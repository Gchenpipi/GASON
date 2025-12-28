import argparse
import json
import logging
import os
import sys
import threading
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from Components.joern_manager import JoernManager
from Components.enhancer import analyze_c_code, get_context, save_context
from Components.slice import merge


class SliceConstructor:
    """
    1. Processes code samples through Joern to extract vulnerability paths.
    2. Enhances extracted paths into readable code snippets.
    3. Stores the results for further analysis.
    """
    
    def __init__(
        self,
        joern_port: int,
        dataset_slice: List[Dict],
        output_path: str,
        log_path: str,
        docker_compose_path: str,
        thread_id: int = 0,
        server_recreation_interval: int = 5,
        max_paths_per_sample: int = 10,
        enhanced_code_output_dir: Optional[str] = None,
    ):
        """
        Initialize a constructor for a specific port and dataset slice.
        
        Args:
            joern_port: Joern server port number.
            dataset_slice: Subset of the dataset to process.
            output_path: Path to the output JSON file for results.
            log_path: Path to the log JSON file for errors and progress.
            docker_compose_path: Path to the Docker Compose YAML file for Joern server management.
            thread_id: ID of the thread running this analyzer instance. Defaults to 0.
            server_recreation_interval: Number of samples to process before recreating the Joern server.
            max_paths_per_sample: Maximum number of vulnerability paths to process per sample.
            enhanced_code_output_dir: Optional directory to save enhanced code snippets.
        """
        self.port = joern_port
        self.dataset_slice = dataset_slice
        self.output_file = output_path
        self.logs_file = log_path
        self.compose_file = docker_compose_path
        self.thread_id = thread_id
        self.thread_name = f"Thread-{thread_id}"
        self.recreate_interval = server_recreation_interval
        self.max_paths = max_paths_per_sample
        self.current_sample = ""
        
        # Create output directories
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
        
        # Create enhanced code directory
        if enhanced_code_output_dir:
            self.enhanced_dir = enhanced_code_output_dir
        else:
            self.enhanced_dir = os.path.join(os.path.dirname(output_path), "enhanced_code")
        Path(self.enhanced_dir).mkdir(parents=True, exist_ok=True)

        # Configure thread-specific logger
        self.logger = logging.getLogger(f"SliceConstruct-{self.thread_id}")
        self.logger.setLevel(logging.INFO)
        
        # The JoernManager will be initialized in process_dataset
        self.joern_manager = None

    def process_dataset(self):
        threading.current_thread().name = f"Analyzer-{self.thread_id}"
        self.logger.info(f"Starting processing of {len(self.dataset_slice)} samples")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.joern_manager = JoernManager(self.port, self.compose_file)

            total_samples = len(self.dataset_slice)


            with tqdm(total=total_samples, desc=f"Thread-{self.thread_id} progress", unit="sample") as pbar:

                for slice_start in range(0, total_samples, self.recreate_interval):
                    slice_end = min(slice_start + self.recreate_interval, total_samples)
                    current_slice = self.dataset_slice[slice_start:slice_end]


                    processed_slice_results = []

                    for sample in current_slice:
                        result_batch = self._process_sample(sample)
                        if result_batch:
                            processed_slice_results.extend(result_batch)
                        pbar.update(1)


                    if processed_slice_results:
                        self._write_batch_results(processed_slice_results)

            self.logger.info("Completed processing all assigned samples")

        except Exception as e:
            self.logger.exception(f"Fatal error in process_dataset: {e}")
            self._write_error_logs(f"Fatal error: {str(e)}")

        finally:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.stop()
            loop.close()

    def _process_sample(self, sample: Dict):
        """
        处理单个样本，不再直接写文件，而是返回处理结果列表。
        """

        self.current_sample = sample["file_name"]

        try:
            self.logger.info(f"Processing sample: {sample['file_name']}")

            file_name = os.path.basename(sample["file_name"])

            joern_results = sample.get("joern_results", {})
            all_paths = joern_results.get("all_paths", [])

            num_flows = len(all_paths)
            self.logger.info(f"Number of flows detected (from JSON): {num_flows}")

            if num_flows == 0:
                self.logger.info("No flows detected, skipping sample")
                return []

            success = joern_results.get("successful_query_validation", True)
            paths = all_paths

            if not success or not paths:
                self.logger.warning("Failed to extract paths or no paths found in JSON")
                return []

            paths_to_process = paths[:min(num_flows, self.max_paths)]
            self.logger.info(f"Processing {len(paths_to_process)} paths out of {num_flows} detected")

            processed_results = self._process_paths(sample, paths_to_process)

            if processed_results:
                self.logger.info(f"Successfully processed {len(processed_results)} paths")
                return processed_results
            else:
                self.logger.warning("No viable paths were processed")
                return []

        except Exception as e:
            self.logger.exception(f"Error processing sample: {e}")
            self._write_error_logs(f"Error processing sample: {str(e)}")
            return []

    def _process_paths(self, sample: Dict, paths: List) -> List[Dict]:
        """
        Process extracted paths to create enhanced code snippets.
        
        Args:
            sample: The original sample data.
            paths: The extracted paths from Joern.
            
        Returns:
            List of processed results including enhanced code snippets.
        """
        results = []
        details = sample.get("details", {})
        source_code = details.get("code", "")
        if not source_code:
            self.logger.error("Missing source code in sample")
            return results
        
        # Create merged paths using the merger functionality
        merged_paths = merge(paths)
        self.logger.info(f"Created {len(merged_paths)} merged paths from {len(paths)} original paths")
        
        # Process each merged path
        for path_idx, merged_path in enumerate(merged_paths):
            try:
                # Analyze code blocks in the source code
                blocks = analyze_c_code(source_code)
                
                # Extract line numbers from the path
                path_line_numbers = [node.get('line_number') for node in merged_path 
                                    if node.get('line_number') is not None]
                
                if not path_line_numbers:
                    self.logger.warning(f"Path {path_idx}: No line numbers found, skipping")
                    continue
                
                # Get context lines that should be included in the enhanced snippet
                context_lines = get_context(path_line_numbers, blocks)
                
                # Create enhanced code file path
                base_name = os.path.basename(sample["file_name"])
                enhanced_file_path = os.path.join(
                    self.enhanced_dir, 
                    f"{base_name}_path{path_idx}_enhanced.c"
                )
                
                # Save the enhanced code to file and get the enhanced code as a string
                enhanced_code = save_context(list(context_lines), source_code, enhanced_file_path)
                
                # Create the result entry
                result = {
                    "dataset": sample.get("dataset", "unknown"),
                    "transformation_idx": sample.get("transformation_idx", 0),
                    "original_file_name": sample.get("original_file_name", ""),
                    "file_name": sample.get("file_name", ""),
                    "llm_queries": sample.get("llm_queries", []),
                    "path_idx": path_idx,
                    "cwe": sample.get("cwe", ""),
                    "label": sample.get("label", ""),
                    "original_code": sample.get("original_code", ""),
                    "code": source_code,
                    "path": merged_path,
                    "context_lines": list(context_lines),
                    "enhanced_code_file": enhanced_file_path,
                    "enhanced_code": enhanced_code
                }
                
                results.append(result)
                self.logger.debug(f"Successfully processed path {path_idx}")
                
            except Exception as e:
                self.logger.exception(f"Error processing path {path_idx}: {e}")
        
        return results

    def _write_batch_results(self, processed_slice_results: list):

        if not processed_slice_results:
            return

        try:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []


            existing_data.extend(processed_slice_results)


            temp_file = f"{self.output_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
            os.replace(temp_file, self.output_file)

            self.logger.info(f"Wrote {len(processed_slice_results)} samples to {self.output_file}")

        except Exception as e:
            self.logger.exception(f"Error writing batch results: {e}")
            self._write_error_logs(f"Error writing batch results: {str(e)}")


    def _write_processed_sample(self, processed_sample: Dict):
        """
        Write a processed sample to the output file.
        
        Args:
            processed_sample: The processed sample data.
        """
        try:
            # Use file lock to prevent race conditions when multiple threads write to the same file
            lock_file = f"{self.output_file}.lock"
            with open(lock_file, 'w') as f:
                f.write(f"Lock created by thread {self.thread_id}")
            
            try:
                # If file exists, read existing data, otherwise start with an empty list
                try:
                    with open(self.output_file, 'r') as f:
                        existing_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    existing_data = []
                
                # Append new sample
                existing_data.append(processed_sample)
                
                # Write back to file (atomic write)
                temp_file = f"{self.output_file}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(existing_data, f, indent=4)
                
                # Rename for atomic replacement
                os.replace(temp_file, self.output_file)
                
                self.logger.info(f"Wrote processed sample to {self.output_file}")
                
            finally:
                # Remove lock file
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                
        except Exception as e:
            self.logger.exception(f"Error writing processed sample: {e}")
            self._write_error_logs(f"Error writing processed sample: {str(e)}")

    def _write_error_logs(self, error_message: str):
        """
        Write error logs to the logs file.
        
        Args:
            error_message: Brief description of the error.
        """
        try:
            # Use file lock to prevent race conditions
            lock_file = f"{self.logs_file}.lock"
            with open(lock_file, 'w') as f:
                f.write(f"Lock created by thread {self.thread_id}")
            
            try:
                # If file exists, read existing data, otherwise start with an empty list
                try:
                    with open(self.logs_file, 'r') as f:
                        existing_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    existing_data = []
                
                # Append new log entry
                existing_data.append({
                    "sample": self.current_sample,
                    "thread_id": self.thread_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": error_message
                })
                
                # Write back to file (atomic write)
                temp_file = f"{self.logs_file}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(existing_data, f, indent=4)
                
                # Rename for atomic replacement
                os.replace(temp_file, self.logs_file)
                
                self.logger.info(f"Wrote logs to file {self.logs_file}")
                
            finally:
                # Remove lock file
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            
        except Exception as e:
            self.logger.error(f"Error writing logs: {e}")


def run_analyzer_thread(
    thread_id: int,
    dataset_slice: List[Dict],
    joern_port: int,
    output_path: str,
    logs_path: str,
    docker_compose_path: str,
    server_recreation_interval: int,
    max_paths_per_sample: int,
    enhanced_code_dir: str,
):
    """
    Run an analyzer in a separate thread.
    
    Args:
        thread_id: ID of the thread.
        dataset_slice: Subset of the dataset to process.
        joern_port: Joern server port number.
        output_path: Path to the output JSON file for results.
        logs_path: Path to the log JSON file for errors and progress.
        docker_compose_path: Path to the Docker Compose YAML file for Joern server management.
        server_recreation_interval: Number of samples to process before recreating the Joern server.
        max_paths_per_sample: Maximum number of vulnerability paths to process per sample.
        enhanced_code_dir: Directory to save enhanced code snippets.
    """
    analyzer = SliceConstructor(
        joern_port=joern_port,
        dataset_slice=dataset_slice,
        output_path=output_path,
        log_path=logs_path,
        docker_compose_path=docker_compose_path,
        thread_id=thread_id,
        server_recreation_interval=server_recreation_interval,
        max_paths_per_sample=max_paths_per_sample,
        enhanced_code_output_dir=enhanced_code_dir,
    )
    
    analyzer.process_dataset()


def distribute_processing(args):
    """
    Distribute the dataset processing across multiple threads.
    
    Args:
        args: Command-line arguments from argparse.
    """
    logger = logging.getLogger("SliceConstructor-Main")
    
    # Load full dataset
    try:
        logger.info(f"Loading dataset from {args.dataset_path}")
        with open(args.dataset_path, 'r') as f:
            dataset = json.load(f)
        logger.info(f"Loaded dataset with {len(dataset)} samples")
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {args.dataset_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from {args.dataset_path}")
        return
    
    # Divide dataset into slices for parallel processing
    slice_size = len(dataset) // args.num_joern_servers
    threads = []
    
    logger.info(f"Starting {args.num_joern_servers} analyzer threads")
    
    # Create and start a thread for each port
    for i in range(args.num_joern_servers):
        # Calculate start and end indices for this thread's slice
        start_idx = i * slice_size
        end_idx = start_idx + slice_size if i < args.num_joern_servers - 1 else len(dataset)
        
        # Create the slice for this thread
        dataset_slice = dataset[start_idx:end_idx]
        
        # Create output and logs file paths
        output_file = os.path.join(args.output_dir, "results", f'thread_{i+1}_results.json')
        logs_file = os.path.join(args.output_dir, "logs", f'thread_{i+1}_logs.json')
        
        # Create directories if they don't exist
        Path(os.path.join(args.output_dir, "results")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.output_dir, "logs")).mkdir(parents=True, exist_ok=True)
        
        # Determine enhanced code output directory for this thread
        thread_enhanced_code_dir = args.enhanced_code_dir if args.enhanced_code_dir else os.path.join(args.output_dir, f'thread_{i+1}_enhanced_code')
        Path(thread_enhanced_code_dir).mkdir(parents=True, exist_ok=True)
        
        # Joern port for this thread
        joern_port = args.base_joern_port + i
        
        logger.info(f"Configuring thread {i+1} with port {joern_port}, processing {len(dataset_slice)} samples")
        
        # Create and start thread
        thread = threading.Thread(
            target=run_analyzer_thread,
            args=(
                i+1,
                dataset_slice,
                joern_port,
                output_file,
                logs_file,
                args.docker_compose_file,
                args.server_recreation_interval,
                args.max_paths_per_sample,
                thread_enhanced_code_dir,
            ),
            name=f"Analyzer-{i+1}"
        )
        thread.start()
        threads.append(thread)
        
        # Brief delay to prevent all threads starting simultaneously
        time.sleep(1)
    
    # Wait for all threads to complete
    for i, thread in enumerate(threads):
        logger.info(f"Waiting for thread {i+1} to complete")
        thread.join()
        logger.info(f"Thread {i+1} completed")
    
    logger.info("All threads completed processing")


def parse_arguments():
    """
    Parse command-line arguments for the vulnerability analyzer.
    
    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Slice Constructor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-d", "--dataset-path", type=str,
        required=True,
        help="Path to the JSON file containing the dataset"
    )
    
    parser.add_argument(
        "-b", "--base-joern-port", type=int,
        default=16240,
        help="Starting port number for the Joern servers"
    )
    
    parser.add_argument(
        "-n", "--num-joern-servers", type=int,
        default=3,
        help="Number of Joern servers to spin up (and threads to create)"
    )
    
    parser.add_argument(
        "-o", "--output-dir", type=str,
        required=True,
        help="Directory to save the processed results"
    )
    
    parser.add_argument(
        "--docker-compose-file", type=str,
        default='docker-compose.yml',
        help="Path to the Docker Compose YAML file for Joern server management"
    )
    
    parser.add_argument(
        "--server-recreation-interval", type=int,
        default=20,
        help="Number of samples to process before recreating each Joern server"
    )
    
    parser.add_argument(
        "--max-paths-per-sample", type=int,
        default=50,
        help="Maximum number of vulnerability paths to process per sample"
    )
    
    parser.add_argument(
        "--enhanced-code-dir",
        type=str,
        default=None,
        help="Optional directory to save enhanced code snippets"
    )
    
    parser.add_argument(
        "--log-level", type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    
    return parser.parse_args()


def setup_logging(log_level):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(threadName)s - %(name)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    """
    Main entry point for the vulnerability analyzer.
    """
    args = parse_arguments()
    setup_logging(args.log_level)
    
    logger = logging.getLogger("Construct_Slice")
    logger.info("Starting Slice Construction")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        distribute_processing(args)
        logger.info("Slice construction completed successfully")
    except Exception as e:
        logger.exception(f"Fatal error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
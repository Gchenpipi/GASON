import json
import os
import sys
import threading
import time
import argparse
import logging
from typing import List, Dict
from Components.model import LLMManager
from Components.joern_manager import JoernManager
import tempfile


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

def gen_query_prompt(code: str):
    instruction = """Your task is to design Precise Joern CPGQL Queries for Vulnerability Analysis.

Objective:
Develop targeted CPGQL Joern queries to:
1. Identify taint flows based on your analysis
2. Capture potential vulnerability paths

Constraints:
- Queries must be executable in Joern/CPGQL
- Use Scala language features for query construction
- Last query must use reachableByFlows to identify vulnerable paths

Output Requirements:
- Provide a JSON object with one field:
  "queries": Sequence of CPGQL queries to detect vulnerability

Expected JSON Output Format:
```json
{
  "queries": ["Query1", "Query2", ..., "Final Reachable Flows Query"]
}
```

Example Output:
```json
{
  "queries": [
    "val freeCallsWithIdentifier = cpg.method.name("(.*_)?free").filter(_.parameter.size == 1).callIn.where(_.argument(1).isIdentifier).l",
    "freeCallsWithIdentifier.flatMap(f => {val freedIdentifierCode = f.argument(1).code; val postDom = f.postDominatedBy.toSetImmutable; val assignedPostDom = postDom.isIdentifier.where(_.inAssignment).codeExact(freedIdentifierCode).flatMap(id => id ++ id.postDominatedBy); postDom.removedAll(assignedPostDom).isIdentifier.codeExact(freedIdentifierCode).reachableByFlows(f.argument(1))}).l"
  ]
}
```
""" 
    prompt = f"{instruction}\nInput:\n{code}"
    return prompt


def gen_fix_prompt(code: str, generated_queries: list, error_info: dict):
    """
    Generate a concise prompt to guide the LLM to fix incorrect Joern CPGQL queries.
    """

    instruction = """Your task is to FIX the previous Joern CPGQL queries that caused syntax or execution errors.

Objective:
1. Analyze the error messages carefully.
2. Correct the incorrect parts of the previous queries.
3. Ensure the corrected queries are valid and executable in Joern/CPGQL.
4. Maintain the original intent — identifying data flows or potential vulnerabilities.

Constraints:
- Use Scala-based CPGQL syntax (compatible with Joern)
- The last query MUST use reachableByFlows to identify vulnerable paths
- Ensure all queries are syntactically valid and logically consistent

Output Requirements:
- Provide a JSON object with one field:
  "queries": Sequence of FIXED and executable CPGQL queries

Expected JSON Output Format:
```json
{
  "queries": ["FixedQuery1", "FixedQuery2", ..., "Final Reachable Flows Query"]
}
```

Example Output:
```json
{
  "queries": [
    "val freeCallsWithIdentifier = cpg.method.name("(.*_)?free").filter(_.parameter.size == 1).callIn.where(_.argument(1).isIdentifier).l",
    "freeCallsWithIdentifier.flatMap(f => {val freedIdentifierCode = f.argument(1).code; val postDom = f.postDominatedBy.toSetImmutable; val assignedPostDom = postDom.isIdentifier.where(_.inAssignment).codeExact(freedIdentifierCode).flatMap(id => id ++ id.postDominatedBy); postDom.removedAll(assignedPostDom).isIdentifier.codeExact(freedIdentifierCode).reachableByFlows(f.argument(1))}).l"
  ]
}
```
"""

    prompt = (
        f"{instruction}\n\n"
        "------\n"
        "Source code under analysis:\n"
        f"{code}\n\n"
        "------\n"
        "Previous (failed) queries:\n"
        f"{json.dumps(generated_queries, indent=2, ensure_ascii=False)}\n\n"
        "------\n"
        "Error information returned by Joern:\n"
        f"{json.dumps(error_info, indent=2, ensure_ascii=False)}\n\n"
        "Note: The next execution will be in a completely clean Joern project. All previous queries, temporary variables, "
        "or project state from prior attempts have been deleted."
        " Do not assume any previous session context. Generate fully self-contained queries. "
        "Please output only the corrected JSON as specified above."
    )

    return prompt

class DatasetProcessor:
    """
    Processes each datapoint, generates the Joern queries for this specific sample, and finally executes the joern queries
    to extract the potentially vulnerable execution paths.
    Each instance runs in its own thread.
    """
    def __init__(self,
                 port: int,
                 dataset_slice: List[Dict],
                 output_file: str,
                 logs_file: str,
                 compose_file: str,
                 llm_model_type: str,
                 llm_endpoint: str,
                 llm_port: int,
                 joern_recreate_interval: int):
        """
        Initialize a dataset processor for a specific port and dataset slice.

        Args:
            port: Joern server port number for this processor.
            dataset_slice: Subset of the dataset to process.
            output_file: Path to the unique output file for this thread's results.
            logs_file: Path to the file to store detailed logs/errors for this thread.
            compose_file: Path to the Docker compose file for Joern.
            llm_model_type: Identifier string for the LLM model type (e.g., "DeepSeek").
            llm_endpoint: URL/path for the LLM service.
            llm_port: Port for the LLM service.
            joern_recreate_interval: Number of samples to process before recreating Joern server.
        """
        self.port = port
        self.dataset_slice = dataset_slice
        self.output_file = output_file
        self.logs_file = logs_file
        self.compose_file = compose_file
        self.joern_recreate_interval = joern_recreate_interval
        self.llm_model_type = llm_model_type
        self.llm_endpoint = llm_endpoint
        self.llm_port = llm_port

        self.current_sample_uuid = ""
        self.sample_log_buffer = []

        # Ensure output/log directories exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.logs_file), exist_ok=True)

    def _log_sample_message(self, level: int, message: str, **kwargs):
        """Logs a message and adds it to the sample buffer."""
        log_msg = f"[Port {self.port} | Sample {self.current_sample_uuid}] {message}"
        logging.log(level, log_msg, **kwargs)
        self.sample_log_buffer.append(f"[{logging.getLevelName(level)}] {message}")

    def _load_processed_files(self) -> set:
        """
        Load the set of already processed file names from the output file.
        """
        processed = set()
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    file_name = item.get("file_name")
                    if file_name:
                        processed.add(file_name)
                logging.info(f"[Port {self.port}] Loaded {len(processed)} processed samples from {self.output_file}")
            except Exception as e:
                logging.warning(f"[Port {self.port}] Could not read processed file list: {e}")
        return processed

    def process_dataset(self, batch_size: int = 50):
        """
        Process the assigned slice of the dataset. Handles Joern server recreation.
        Collects samples in batches before writing results to file.
        """
        import nest_asyncio
        import asyncio
        from Components.model import Model
        from tqdm import tqdm

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        nest_asyncio.apply(loop)

        self.joern_manager = JoernManager(self.port, self.compose_file)
        model_enum = Model[self.llm_model_type.upper()]
        self.llm_manager = LLMManager(model_enum, self.llm_endpoint, port=self.llm_port)

        thread_name = threading.current_thread().name
        active_joern_project = None
        batch_results = []

        try:
            num_samples = len(self.dataset_slice)
            with tqdm(total=num_samples, desc=f"{thread_name} (Port {self.port})",
                      position=self.port % 20, leave=True) as pbar:

                processed_files = self._load_processed_files()
                self._log_sample_message(logging.INFO, f"Skipping {len(processed_files)} already processed samples")

                for i, sample in enumerate(self.dataset_slice):
                    file_name = sample.get("file_name", "").split("/")[-1]
                    if file_name in processed_files:
                        self._log_sample_message(logging.INFO, f"Skipping already processed file: {file_name}")
                        pbar.update(1)
                        continue

                    self.current_sample_uuid = sample.get("uuid", "N/A")
                    self.sample_log_buffer = []
                    active_joern_project = None

                    if i % self.joern_recreate_interval == 0:
                        if active_joern_project:
                            try:
                                self.joern_manager.delete_project(active_joern_project)
                            except Exception as cleanup_e:
                                self._log_sample_message(logging.WARNING,
                                                         f"Failed cleanup before recreate: {cleanup_e}")
                            finally:
                                active_joern_project = None

                        self._log_sample_message(logging.INFO,
                                                 f"Processed {self.joern_recreate_interval} samples. Recreating Joern server.")
                        is_healthy = self.joern_manager.recreate_server()
                        if not is_healthy:
                            self._log_sample_message(logging.ERROR,
                                                     "Joern server unhealthy after recreation. Exiting thread.")
                            self._write_error_logs()
                            return
                        self._log_sample_message(logging.INFO, "Joern server recreated successfully.")


                    try:
                        self._log_sample_message(logging.INFO, "Starting processing.")
                        if not file_name:
                            raise ValueError("Invalid file_name in sample")

                        result = self._process_single_sample(sample, file_name)
                        if result:
                            batch_results.append(result)
                            active_joern_project = file_name
                            self._log_sample_message(logging.INFO, "Sample processed successfully.")
                        else:
                            self._log_sample_message(logging.WARNING, "Processing failed, see logs.")
                            self._write_error_logs()
                            active_joern_project = file_name


                        if len(batch_results) >= batch_size:
                            self._write_processed_batch(batch_results)
                            batch_results.clear()
                            self._log_sample_message(logging.INFO, f"Batch of {batch_size} samples written to file.")

                    except Exception as e:
                        self._log_sample_message(logging.ERROR, f"Unhandled error processing sample: {e}",
                                                 exc_info=True)
                        self._write_error_logs()
                        if 'file_name' in locals() and file_name:
                            active_joern_project = file_name

                    finally:
                        if active_joern_project:
                            try:
                                self.joern_manager.delete_project(active_joern_project)
                            except Exception as cleanup_e:
                                self._log_sample_message(logging.WARNING,
                                                         f"Failed to cleanup Joern project {active_joern_project}: {cleanup_e}")
                            finally:
                                active_joern_project = None

                    pbar.update(1)


                if batch_results:
                    self._write_processed_batch(batch_results)
                    batch_results.clear()
                    self._log_sample_message(logging.INFO, "Final batch written to file.")

        except Exception as e:
            logging.error(f"[Port {self.port}] Critical error in thread execution: {e}", exc_info=True)
            self.sample_log_buffer.append(f"[CRITICAL] Thread execution error: {e}")
            self._write_error_logs()
        finally:
            loop.close()

    def _process_single_sample(self, sample: Dict, file_name: str) -> Dict | None:
        """
        Process a single sample using Joern and LLM, with LLM-based retry correction.
        """
        MAX_LLM_RETRIES = 3  # 最大重试次数

        try:

            required_keys = ["file_name", "code", "label"]
            if not all(key in sample for key in required_keys):
                missing = [key for key in required_keys if key not in sample]
                raise ValueError(f"Sample missing required keys: {missing}")
            code_content = sample["code"]


            self._log_sample_message(logging.DEBUG, f"Loading project '{file_name}' into Joern.")
            stdout = self.joern_manager.load_project(file_name)

            if "io.joern.console.ConsoleException" in stdout:
                self._log_sample_message(logging.ERROR,
                                         f"Joern ConsoleException loading project '{file_name}': {stdout}")
                raise Exception(f"Joern ConsoleException loading project: {stdout}")
            elif "fail" in stdout.lower() or "error" in stdout.lower():
                self._log_sample_message(logging.WARNING,
                                         f"Potential issue loading project '{file_name}' in Joern: {stdout}")


            self._log_sample_message(logging.DEBUG, "Generating CPGQL queries using LLM.")
            prompt = gen_query_prompt(code_content)
            message_history = [{"role": "user", "content": prompt}]

            llm_response = self.llm_manager.send_messages(message_history)
            if not llm_response:
                raise Exception("LLM response was empty or invalid.")

            completion_text = self.llm_manager.get_completion_text(llm_response)
            if not completion_text:
                raise Exception("Failed to extract completion text from LLM response.")

            llm_answer = self.llm_manager.extract_queries(completion_text)
            if not llm_answer or "queries" not in llm_answer or not isinstance(llm_answer["queries"], list):
                raise Exception("Failed to extract valid queries list from LLM response.")

            generated_queries = llm_answer["queries"]
            self._log_sample_message(logging.INFO, f"LLM generated {len(generated_queries)} queries.")
            self._log_sample_message(logging.DEBUG, f"Generated Queries: {generated_queries}")


            attempt = 0
            successful = False
            paths = []
            last_error_info = None

            while attempt <= MAX_LLM_RETRIES:
                self._log_sample_message(logging.INFO,
                                         f"Running queries in Joern (attempt {attempt + 1}/{MAX_LLM_RETRIES})")


                if attempt > 0:
                    self._log_sample_message(logging.DEBUG, f"Cleaning up old project before retry #{attempt}.")
                    self.joern_manager.delete_project(file_name)
                    self._log_sample_message(logging.DEBUG, f"Reloading project '{file_name}' for retry #{attempt}.")
                    self.joern_manager.load_project(file_name)

                successful, result = self.joern_manager.run_queries(generated_queries, code_content)

                if successful:
                    paths = result
                    break


                last_error_info = result
                self._log_sample_message(logging.WARNING, f"Joern query failed (attempt {attempt + 1}): {result}")

                attempt += 1
                if attempt > MAX_LLM_RETRIES:
                    break

                fix_prompt = gen_fix_prompt(
                    code=code_content,
                    generated_queries=generated_queries,
                    error_info=result
                )

                fix_history = [{"role": "user", "content": fix_prompt}]
                llm_fix_response = self.llm_manager.send_messages(fix_history)

                completion_text = self.llm_manager.get_completion_text(llm_fix_response)
                llm_fixed = self.llm_manager.extract_queries(completion_text)

                if not llm_fixed or "queries" not in llm_fixed:
                    self._log_sample_message(logging.ERROR, "LLM failed to fix queries, stopping retry.")
                    break

                generated_queries = llm_fixed["queries"]
                self._log_sample_message(logging.INFO, f"LLM regenerated {len(generated_queries)} corrected queries.")


            sample_result = {
                "file_name": sample["file_name"],
                "llm_model_type": self.llm_model_type,
                "llm_queries": generated_queries,
                "label": sample["label"],
                "joern_results": {
                    "all_paths": paths,
                    "successful_query_validation": successful
                },
                "processing_status": "success" if successful else "failed_after_retries",
                "details": sample
            }


            if not successful and last_error_info:
                sample_result["joern_results"]["error_info"] = last_error_info

            return sample_result

        except Exception as e:
            self._log_sample_message(logging.ERROR, f"Error processing sample: {e}", exc_info=True)
            return None

    def _safe_write_json(self, data, output_path):
        """安全写入JSON文件（写入临时文件后原子替换）"""
        dir_name = os.path.dirname(output_path)
        os.makedirs(dir_name, exist_ok=True)


        fd, tmp_path = tempfile.mkstemp(dir=dir_name)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
                json.dump(data, tmp_file, indent=4, ensure_ascii=False)
            os.replace(tmp_path, output_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e

    def _write_processed_sample(self, processed_sample: dict):
        """Appends a processed sample to the thread-specific JSON output file safely."""
        try:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    self._log_sample_message(logging.WARNING, f"Output file {self.output_file} invalid JSON. Overwriting.")
                    existing_data = []
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []

            existing_data.append(processed_sample)


            self._safe_write_json(existing_data, self.output_file)

            logging.info(f"[Port {self.port}] Appended result for sample {self.current_sample_uuid} safely to {self.output_file}")

        except Exception as e:
            logging.error(f"[Port {self.port} | Sample {self.current_sample_uuid}] Error writing processed sample safely: {e}", exc_info=True)
            self.sample_log_buffer.append(f"[ERROR] Safe write failed: {e}")
            self._write_error_logs()


    def _write_processed_batch(self, batch_results: list):
        """Writes a batch of processed samples to the output file in one atomic operation."""
        try:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []

            existing_data.extend(batch_results)


            self._safe_write_json(existing_data, self.output_file)

            logging.info(f"[Port {self.port}] Safely wrote batch of {len(batch_results)} samples to {self.output_file}")

        except Exception as e:
            logging.error(f"[Port {self.port}] Error safely writing batch to {self.output_file}: {e}", exc_info=True)
            self.sample_log_buffer.append(f"[ERROR] Failed safe batch write: {e}")
            self._write_error_logs()


    def _write_error_logs(self):
        """Appends the buffered logs for the current sample to the thread's log file."""
        if not self.sample_log_buffer:
            return

        try:
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "port": self.port,
                "sample_uuid": self.current_sample_uuid,
                "log_messages": self.sample_log_buffer
            }


            try:
                with open(self.logs_file, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
                if not isinstance(existing_logs, list):
                     logging.warning(f"Log file {self.logs_file} corrupt. Starting new list.")
                     existing_logs = []
            except (FileNotFoundError, json.JSONDecodeError):
                existing_logs = []


            existing_logs.append(log_entry)


            with open(self.logs_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, indent=4)

            logging.info(f"[Port {self.port}] Appended logs for sample {self.current_sample_uuid} to {self.logs_file}")

        except Exception as e:
            logging.critical(f"[Port {self.port} | Sample {self.current_sample_uuid}] CRITICAL ERROR: Failed to write to error log file {self.logs_file}: {e}", exc_info=True)
        finally:

             self.sample_log_buffer = []


def main():
    """
    Main function to parse arguments, set up, and run dataset processing threads.
    """
    parser = argparse.ArgumentParser(description="Process a dataset of code samples using Joern and an LLM in parallel.")


    parser.add_argument("-d", "--dataset-file", type=str, required=True,
                        help="Path to the input dataset JSON file.")
    parser.add_argument("-o", "--output-base-dir", type=str, required=True,
                        help="Base directory for output files. 'results' and 'logs' subdirectories will be created here.")
    parser.add_argument("-c", "--compose-file", type=str, default="docker-compose.yml",
                        help="Path to the Docker Compose file for Joern servers.")
    parser.add_argument("-n", "--num-workers", type=int, default=3,
                        help="Number of parallel threads/workers (and Joern instances) to use.")
    parser.add_argument("--base-joern-port", type=int, default=16240,
                        help="Starting port number for Joern servers.")
    parser.add_argument("--joern-recreate-interval", type=int, default=8,
                        help="Number of samples to process before recreating a Joern server instance.")


    parser.add_argument("--llm-model-type", type=str, choices=["vLLM", "DeepSeek", "Ollama"], default="Ollama", # Default based on original code
                        help="Identifier string for the type of LLM model to use (e.g., 'vLLM', 'DeepSeek'). Passed to LLMManager.")
    parser.add_argument("--llm-endpoint", type=str, default='LLMxCPG-Q-Q4_K_M',
                        help="Endpoint URL or path for the LLM service (e.g., '/path/to/model' or 'http://host:port').")
    parser.add_argument("--llm-port", type=int, default=37183,
                        help="Port number for the LLM service.")

    args = parser.parse_args()


    if not os.path.isfile(args.dataset_file):
        logging.error(f"Dataset file not found: {args.dataset_file}")
        sys.exit(1)
    if args.num_workers <= 0:
        logging.error("--num-workers must be positive.")
        sys.exit(1)



    results_dir = os.path.join(args.output_base_dir, "results")
    logs_dir = os.path.join(args.output_base_dir, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    logging.info(f"Outputs will be saved under: {args.output_base_dir}")
    logging.info(f"Number of workers: {args.num_workers}")



    try:
        with open(args.dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if not isinstance(dataset, list):
             raise TypeError("Dataset file does not contain a valid JSON list.")
        logging.info(f"Loaded dataset with {len(dataset)} samples from {args.dataset_file}")
    except (json.JSONDecodeError, TypeError, FileNotFoundError, OSError) as e:
        logging.error(f"Failed to load or parse dataset file {args.dataset_file}: {e}")
        sys.exit(1)


    if not dataset:
        logging.warning("Dataset is empty. Exiting.")
        sys.exit(0)


    dataset = dataset[:]

    slice_size = len(dataset) // args.num_workers
    remainder = len(dataset) % args.num_workers
    threads = []
    start_idx = 0


    logging.info("Starting worker threads...")
    for i in range(args.num_workers):

        current_slice_size = slice_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_slice_size

        if start_idx >= len(dataset):
            logging.warning(f"Worker {i+1} has no data assigned, reducing effective worker count.")
            continue

        dataset_slice = dataset[start_idx:end_idx]

        output_file = os.path.join(results_dir, f'thread_{i+1}_results.json')
        logs_file = os.path.join(logs_dir, f'thread_{i+1}_logs.json')
        port = args.base_joern_port + i


        processor = DatasetProcessor(
            port=port,
            dataset_slice=dataset_slice,
            output_file=output_file,
            logs_file=logs_file,
            compose_file=args.compose_file,
            llm_model_type=args.llm_model_type,
            llm_endpoint=args.llm_endpoint,
            llm_port=args.llm_port,
            joern_recreate_interval=args.joern_recreate_interval
        )

        thread = threading.Thread(target=processor.process_dataset, name=f"Worker-{i+1}")
        thread.start()
        threads.append(thread)
        logging.info(f"Started Worker-{i+1} (Port {port}) processing {len(dataset_slice)} samples. Output: {output_file}, Logs: {logs_file}")

        start_idx = end_idx


    for thread in threads:
        thread.join()

    logging.info("All worker threads have completed processing.")


if __name__ == "__main__":
    main()

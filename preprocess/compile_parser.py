from tree_sitter import Language

Language.build_library(

  # Store the library in the `build` directory
  'build/my-languages.so',

  # Include one or more languages
  [
    'vendor/tree-sitter-c-0.23.4',
    'vendor/tree-sitter-cpp-master'
    # 'treesitter/tree-sitter-java',
    # 'treesitter/tree-sitter-python',
    # 'treesitter/tree-sitter-cpp',
  ]
)

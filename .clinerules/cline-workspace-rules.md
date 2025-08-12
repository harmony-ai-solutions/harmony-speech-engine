# Project Guidelines

## Workspace Definitions
- Folders of Relevant Projects and repos have been added to the Workspace file.
- The location of the current Workspace File is stored inside the env var HARMONY_AI_VSC_WORKSPACE_FILE.
- If the env var is not set or the workspace file is unavailable, always ask for its location at the beginning of a task.
- Relative paths in the workspace file are always relative to the Workspace file location.
- In case a project or module that is being referenced in the instruction has not been added the workspace file, ask for guidance.

## Agent editing rules
- If multiple files have similar naming, make sure to always re-read a file before an edit, so you have the correct content.

# Building the project
- Do not attempt to build the project to test for compilation issues. Building and Tests will be performed manually after fully finishing the task.

## Memory Bank usage
- Each Repository has it's own memory bank with context about itself and it's role within the project, located in a folder named memory-bank.
- When evaluating a task, always read the memory bank of each project repository that is part of the task so you're up to date.
- When updating the memory bank, always inspect the memory banks of all repositories which have been touched while working on the task whether updates are needed.
- When updating the memory bank, also check whether the docs and changelog of the repository require updates. 

## Documentation Requirements
- Update relevant documentation in /docs when modifying features
- Keep README.md in sync with new capabilities
- Maintain changelog entries in CHANGELOG.md for changes which are relevant for users.
- Do not include implementation details like names of constants or code files in the changelog.


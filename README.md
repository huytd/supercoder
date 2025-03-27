# SuperCoder

Welcome to SuperCoder! A coding agent that runs in your terminal.

![image](https://github.com/user-attachments/assets/201d7b80-9fa0-43f0-a775-46fe7640ee5e)

## Features

SuperCoder equips you with an array of powerful tools to simplify your development workflow. It offers the following features:

- Code Search: Performs complex code searches across your project to quickly locate specific patterns.
- Project Structure Exploration: Provides an organized view of your project's folders and files, making navigation a breeze.
- Code Editing: Enables you to modify your codebase seamlessly with natural language commands.
- Bug Fixing: Automatically fixes bugs and implements improvements based on your detailed requests.
- Cursor Rules Support: Leverages Cursor Rules to intelligently understand and modify your code at precise locations.

## Installation

To install SuperCoder as a native binary, execute the following command:

```bash
sbt universal:packageBin
```

This command will compile and package the project into a binary file located at `target/universal/supercoder-0.1.0-SNAPSHOT.zip`. Extract this file to access the binary in the `bin` directory.

Copy the `supercoder` file to somewhere in your `PATH` to run the agent from anywhere in your terminal.

## Usage

### Configure the Agent

#### Option 1: Using OpenAI API
Before running the agent, you need to have the `OPENAI_API_KEY` environment variable configured. You can obtain an API key by signing up at [OpenAI](https://platform.openai.com/).

```shell
export OPENAI_API_KEY=<API_KEY>
export OPENAI_MODEL=<MODEL> # default to "o3-mini", so watch your wallet
```

#### Option 2: Using Local Models or any OpenAI-compatible API
If you have a local model or any other OpenAI-compatible API, you can configure SuperCoder to use it, by setting the following environment variables:

```shell
export SUPERCODER_BASE_URL=<URL>
export SUPERCODER_API_KEY=<URL>
export SUPERCODER_MODEL=<URL>
```

Note that, if you are using Google Gemini, you will need to set `SUPERCODER_GEMINI_MODE=true` as well.

It's important to note that the model you are using should support tools calling.

### Running the Coding Agent

After building the project, extract and run the generated binary. Once running, you can type natural language commands such as:

- "Search for usage of function XYZ"
- "Edit file path/to/file.scala to add a new method"
- "Show me the project structure"

The agent will interpret your commands and invoke the appropriate tool.

### Interacting with the Tools

SuperCoder supports the following tools:

- **CodeSearchTool**: Helps in searching for specific code patterns across the project.
- **CodeEditTool**: Allows editing of files within the project.
- **FileReadTool**: Reads and displays file content.
- **ProjectStructureTool**: Provides an overview of the project folders and files.
- **CodeExecutionTool**: Executes shell commands based on the agent's assessment.

## Development

For development purposes, follow these instructions to set up your environment:

### Prerequisites

- Java 8 or above
- SBT (Scala Build Tool)

### Setup

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd SuperCoder
    ```

2. Build the project using SBT:
    ```bash
    sbt compile
    ```

3. Run tests to ensure everything is working as expected:
    ```bash
    sbt test
    ```

## Contributing

Contributions, issues, and feature requests are welcome! Please check the [issues page](issues) if you want to contribute.

## License

This project is open source and available under the MIT License.

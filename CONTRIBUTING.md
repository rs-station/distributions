# Contributing to `rs-distributions`

We actively welcome contributions to `rs-distributions`.
You can help contribute by filing issues on the [GitHub repository](https://github.com/rs-station/distributions/issues), adding distributions, fix outstanding issues, and by contributing with examples or to the documentation.

The following sections provide guidelines for setting up a development environment and explain how the steps to follow before submitting a pull request.
## Getting Started

1. [Fork the repository on GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
2. Clone your forked repository to your local machine and setup a development environment.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository.
5. [Submit a pull request to the main repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests)

## Development setup

To setup a development environment, ensure you have Python 3.8 or higher.
You will also need the Hatch package manger.
Follow these steps to setup a development environment:

1. [Install Hatch package manager](https://hatch.pypa.io/latest/install/).
   Example installation with pip:
 ```bash
 pip install hatch
 ```

2. In your shell, go to the project directory and install the project dependencies:

```bash
hatch env create
```

3. Activate the development environment:

```bash
hatch shell
```

4. Confirm the project is installed:

```bash
pip show rs-distributions
```

## Running Tests

We use [pytest](https://docs.pytest.org/en/8.0.x/) for running tests.
To run all of the tests, use the following command:

```bash
hatch run test
```

You can also run individual test files.
This will run all of the tests in `path/to/test_file.py`:
```bash
hatch run test path/to/test_file.py
```

Additionally, you can run a specific test class within a test file.
This will run `path/to/test_file.py::TestClassName`
```bash
hatch run test path/to/test_file.py::TestClassName
```

## Continuous Integration

The tests for `rs-distributions` will automatically run upon any pull request.
Any issues will need to be fixed before the changes are merged into the main repository.

## Code Style and Formatting

We follow the [Black](https://black.readthedocs.io/en/stable/the_black_code_style/index.html) code style and use [Ruff](https://github.com/astral-sh/ruff) for linting.
To check for errors in code formatting, use the following command:

```bash
hatch fmt --check
```

To automatically format your code, use:

```bash
hatch fmt
```

Please ensure that your code passes the formatting checks before submitting a pull request.

## Submitting Pull Requests

When you are ready to submit a pull request, please ensure that:
- Your changes are based on the latest main branch.
- Your code follows the project's code style and formatting guidelines.
- Your commit messages are descriptive and explain the purpose of your changes.
- Your pull request description provides a clear explanation of your changes.


## Building Documentation Locally

We use [MkDocs](https://www.mkdocs.org) for building the project documentation.
To build the documentation locally, you can use the following command:
```bash
hatch run docs:build
```

To preview the documentation locally, you can use:
```batch
hatch run docs:serve
```

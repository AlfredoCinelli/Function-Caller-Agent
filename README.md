# 🔍🤖 WebExplorer Agent

An intelligent agent that searches and synthesizes information from the web using Tavily Search API, Google Serper and Wikipedia, presented through a sleek Streamlit interface.

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)

## ✨ Features

- **Dual Search Capabilities**: Leverage both Tavily's AI-powered (and Google Serper) search and Wikipedia's vast knowledge base
- **Smart Results Synthesis**: Combines and summarizes information from multiple sources
- **Interactive UI**: Built with Streamlit for a seamless user experience
- **Context-Aware Responses**: Maintains conversation history for more relevant follow-ups
- **Full logging**: Keeps a record of all interactions for debugging and analysis
- **Full tracing**: Logs every interaction, including the user's query, the agent's response, and the sources used via [LangSmith](https://www.smith.langchain.com)
- **Easy Settingup**: Use [uv](https://docs.astral.sh/uv/) to  set up the project environment

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/AlfredoCinelli/ReAct-Agent
cd repo_name
```

2. Install dependencies ([install uv first](https://docs.astral.sh/uv/getting-started/installation/)):
```bash
uv sync
```

3. Set up your environment variables (in the `local/.env` file), see below for more details.

4. Run the application (via Make):
```bash
make app
```

## 🛠️ Technical Architecture

```
repo/
├── app.py               # Streamlit application entry src
├── src/
│   ├── backend.py       # Main module with the agent
│   ├── utils/           # Main package with utils modules
│   │   ├── callbacks.py # Module for callbacks
│   │   ├── logging.py   # Module for logging
│   │   ├── tools.py     # Module defining tools
│   │   └── misc.py # Module with helper functions
├── chatbot.py # First version of the chatbot
```

## 📝 Configuration

Configure the agent through `local/.env` file:

```yaml
TAVILY_API_KEY="your_tavily_api_key"
SERPER_API_KEY="your_serpapi_api_key"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langchain_api_key"
LANGCHAIN_PROJECT="your_project_name"
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Tavily API](https://tavily.com) for powerful search capabilities
- [Serper API](https://serper.dev) for advanced search capabilities
- [Wikipedia API](https://pypi.org/project/wikipedia/) for knowledge base access
- [Streamlit](https://streamlit.io) for the amazing UI framework

## 📞 Support

- 📧 Email: alfredocinelli@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/AlfredoCinelli/ReAct-Agent/issues)

---
Made with ❤️ by [Alfredo Cinelli](https://github.com/alfredocinelli)
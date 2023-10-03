abinbev-llm
==============================

This project aims to build an MVP to solve the following problem statement:

**Build a Chatbot to perform Generative QA with indexed documents in a vector database as knowledge base**

https://llm-chatbot-rbjv6s9pz7kcjrjwd4clwm.streamlit.app

Example of usage: How can i estimate rainfall?

Also works in spanish: Como puedo estimar la caida de lluvia?


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- WIP
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├──connectors          <- Data Connectors to external sources
    │
    ├── docs               <- WIP
    │
    ├── authentication     <- Authentication implementation for streamlit appication
    │
    ├── pages              <- Pages of the streamlit application 
    │
    ├── config             <- Configuration file
    │
    ├── logs               <- WIP
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries (WIP)
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials. (WIP)
    │
    ├── reports            <- WIP
    │   └── figures        <- WIP
    │
    ├── requirements.txt (requirements_dev.txt)   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── static             <- Static files to be used inside the app
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │    └── parser.py Parser for files
    │   │  
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── client_config.toml <- Customizable file for client
    │
    ├── settings.py        <- File for setting paths
    │
    ├── streamlit_app.py   <- Streamlit applicatin base
    │
    ├── streamlit_app_pages.py <- Streamlit application splitted into pages
    │
    ├── test_environment.py <- WIP
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

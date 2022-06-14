# create directory to store the credentials and config files
mkdir -p ~/.streamlit/
# add config details
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
linters:
	@sh bash/execute_linters.sh $(path)

app:
	streamlit run app.py

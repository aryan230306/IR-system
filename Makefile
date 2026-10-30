.PHONY: install test serve bench demo docker
install:
	pip install -r requirements.txt
test:
	python -m pytest -q
serve:            ## live synthetic news stream + dashboard on http://localhost:8000
	python -m streamir serve --rate 40
bench:
	python -m streamir bench --n 20000
demo:             ## quiet server, push your own docs via the API
	python -m streamir serve --source none
docker:
	docker build -t streamir . && docker run -p 8000:8000 streamir

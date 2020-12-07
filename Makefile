CHECKSTYLE_CMD = flake8 --max-line-length 120 --extend-ignore W605

checkstyle:
	$(CHECKSTYLE_CMD) *.py */*.py
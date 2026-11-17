from parsing.location._cli import run
from .inference import LocationFilter

if __name__ == "__main__":
    run(LocationFilter, "filter")

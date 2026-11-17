from parsing.location._cli import run
from .inference import LocationRewriter

if __name__ == "__main__":
    run(LocationRewriter, "rewrite")

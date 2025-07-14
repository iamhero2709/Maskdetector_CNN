# serving/flask_app.py
from flask import Flask, render_template, request, redirect, url_for
import requests, os

# Local dev -> FastAPI at 127.0.0.1; Docker Compose -> hostname "api"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img = request.files["image"]
        if not img:
            return redirect(request.url)

        resp = requests.post(
            API_URL,
            files={"file": (img.filename, img.read(), img.content_type)},
        )

        if resp.ok:
            return render_template("index.html", prediction=resp.json())
        return f"Error: {resp.text}", resp.status_code

    return render_template("index.html", prediction=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

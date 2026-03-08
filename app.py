from flask import Flask, send_file

app = Flask(__name__)

@app.route("/")
def home():
    return "EV Battery API Running"

@app.route("/dashboard")
def dashboard():
    return send_file("dashboard.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
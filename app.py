from flask import Flask, request, jsonify
from recommendation_engine import recommend

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def get_recommendation():
    user_data = request.json.get('user_data')
    if user_data:
        recommendation = recommend(user_data)
        return jsonify({"recommended_content_id": recommendation[0]})
    return jsonify({"error": "No user data provided"}), 400

if __name__ == "__main__":
    app.run(debug=True)

import json
import urllib.request
import urllib.parse

def test_endpoints():
    base = "http://localhost:5000"

    # Test home
    with urllib.request.urlopen(f"{base}/") as response:
        print(f"GET / -> Status: {response.status}")

    # Test demos
    demos = ["real", "deepfake", "lipsync", "low_quality"]
    for d in demos:
        data = urllib.parse.urlencode({"demo_choice": d}).encode("utf-8")
        req = urllib.request.Request(f"{base}/api/analyze", data=data)
        with urllib.request.urlopen(req) as resp:
            data_json = json.loads(resp.read().decode("utf-8"))
            print(f"Demo '{d}' => Verdict: {data_json['verdict']}, Risk: {data_json['risk_score']}%, Clues: {len(data_json['clues'])}")

if __name__ == "__main__":
    test_endpoints()

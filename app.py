from flask import Flask, render_template, request
import model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('firstpage.html')

@app.route('/process', methods=['POST'])
def process():
    airways = request.form['Air_Ways']
    source_city=request.form['source_city']
    destination_city=request.form['destination_city']
    departure_time=request.form['departure_time']
    arrival_time=request.form['arrival_time']
    class_type=request.form['class']
    stops=request.form['stops']
    # duration=request.form['duration']
    days_left=request.form['days']

    result=model.model_prediction(airways,source_city,destination_city,departure_time,arrival_time,class_type,stops,days_left)
    start=source_city
    end=destination_city
    return render_template('firstpage.html',result=result,start=start,end=end)

if __name__ == '__main__':
    app.run(debug=True)

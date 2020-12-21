from flask import Flask, request, redirect, render_template, jsonify
import webbrowser
import json
from api import api_responses
from db import db_helpers
import boto3
import botocore
from datetime import datetime
from flask_socketio import SocketIO, send
import time
from threading import Thread
from secrets_cloud import Secrets_cloud


app = Flask(__name__)
'''
socketio = SocketIO(app)
'''

# AWS Credentials
aws_access_key = Secrets_cloud.get_aws_access_key() #os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = Secrets_cloud.get_aws_secret_access_key() #os.environ['AWS_SECRET_ACCESS_KEY']
dynamo = boto3.resource('dynamodb', region_name='eu-central-1', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_access_key)

@app.route("/")
def index():
    return render_template('Dashboard.html')


@app.route("/register/<string:cart_id>")
def registration_code(cart_id):
    return render_template('registration.html')


@app.route("/register/<string:cart_id>/<string:cart_code>")
def registration_code_check(cart_id, cart_code):
    db_cart_code = db_helpers.get_cart_code(cart_id)
    if cart_code == db_cart_code:
        return redirect("/register/activate/"+cart_id+"/"+cart_code)
    else:
        print(cart_id)
        print(cart_code)
        return render_template('SessionExpired.html')


@app.route("/register/activate/<string:cart_id>/<string:cart_code>")
def activate(cart_id, cart_code):
    #return render_template('scale_activity.html')
    current_time = datetime.utcnow()
    while True:
        last_activity_str = str(db_helpers.get_last_activity(cart_id))
        last_activity = datetime.strptime(last_activity_str, '%Y-%m-%d %H:%M:%S.%f')
        time_diff_sec = (current_time - last_activity).total_seconds()
        if abs(time_diff_sec) < 10:
            new_cart_session = str(hash(cart_id + cart_code + str(datetime.utcnow())))
            db_helpers.set_cart_session(cart_id, new_cart_session)
            return redirect("/cartid/"+cart_id+"/"+new_cart_session)
        elif (datetime.utcnow() - current_time).total_seconds() > 10:
            return render_template('SessionExpired.html')


# test of dynamic content within the dashboard
@app.route("/_stuff/<string:cart_id>", methods=['GET'])
def stuff(cart_id):
    print("+++++"+cart_id)
    unsorted_list = db_helpers.get_products(cart_id)
    print("UNsorted list:")
    print(unsorted_list)

    print("sorted list:")
    sorted_list = db_helpers.sort(unsorted_list)
    print(sorted_list)

    return jsonify(result=sorted_list)


'''
This is the heart and soul of the application: The main UI for the shopping cart
'''
@app.route("/cartid/<string:cart_id>/<string:cart_session>")
def cart_ui(cart_id, cart_session):
    db_cart_session = db_helpers.get_cart_session(cart_id)
    if db_cart_session == cart_session:
        list = db_helpers.get_products("1")
        print(list)
        products = db_helpers.get_products(cart_id)
        #Thread(target=handleProducts).start()
        return render_template('Dashboard_test.html', cart_id=cart_id, cart_session=cart_session)
    else:
        return "ERROR"


'''
This is the webpage for the checkout after purchasing
'''
@app.route("/Feedback")
def Feedback():
    return render_template('Feedback.html')


@app.route("/AboutUs")
def AboutUs():
    return render_template('AboutUs.html')


@app.route("/Dashboard")
def Dashboard():
    return render_template('Dashboard.html')


@app.route("/Registration")
def Registration():
    return render_template('registration.html')

@app.route("/paypage")
def PayPage():
    return render_template('PayPage.html')

@app.route("/paypage/<string:cart_id>_<string:cart_session>")
def paypage(cart_id, cart_session):
    print("Button wurde gedrückt")
    print(cart_id)
    print(cart_session)
    db_helpers.push_to_archive(cart_id, cart_session)
    return render_template('PayPage.html')

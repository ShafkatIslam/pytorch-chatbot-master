import random
import json

import printjson as printjson
from flask import Flask, render_template, url_for, request, session, redirect
from flask_pymongo import PyMongo
import bcrypt
from flask_bcrypt import Bcrypt
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

from flask import Flask, render_template, request, url_for, flash, redirect, session

app = Flask(__name__)
app.secret_key = "super secret key"
bcrypt = Bcrypt(app)

'''app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'
sess = Session()'''

app.config['MONGO_URI'] = 'mongodb+srv://Shafkat:Shafkaait@cluster0.mkcss.mongodb.net/Chatbot'
#app.config['MONGO_URI'] = 'mongodb+srv://Shafkat:shafkaait@cluster0.3z9yl.mongodb.net/Chatbot'

mongo = PyMongo(app)

@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html',message= session['username'])
    return render_template('mainindex.html')


@app.route('/login', methods=['POST'])
def login():
    usernames = request.form['username']
    passwords = request.form['pass']
    if usernames is None or usernames == '':
        return render_template('mainindex.html', error='Username is empty')
    if passwords is None or passwords == '':
        return render_template('mainindex.html', error='Password is empty')
    else:
        users = mongo.db.users
        login_user = users.find_one({'name': request.form['username']})
        all_questions = []
        all_answers = []
        if login_user:
            if bcrypt.check_password_hash(login_user['password'], request.form['pass']):
                session['username'] = request.form['username']
                chatTable = mongo.db.Chat
                chatData = chatTable.find({'name': request.form['username']})
                print(chatData)
                if chatData:
                    for x in chatData:
                        questions = x['details']['question']
                        answers = x['details']['answer']
                        all_questions.append(questions)
                        all_answers.append(answers)

                    return render_template('index.html', message=session['username'],
                                           all_history=zip(all_questions, all_answers))
                else:
                    return render_template('index.html', message=session['username'],
                                           all_history=zip(all_questions, all_answers))

        return render_template('mainindex.html', error='Invalid username/password combination')

@app.route('/logout',methods=['GET', 'POST'])
def logout():
    if request.method == 'POST':
        session.pop('username', None)
        return render_template('mainindex.html')
    return render_template('mainindex.html')

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        usernames = request.form['username']
        passwords = request.form['pass']
        if usernames is None or usernames == '':
            return render_template('register.html', error='Username is required')
        if passwords is None or passwords == '':
            return render_template('register.html', error='Password is required')
        else:
            users = mongo.db.users
            existing_user = users.find_one({'name': request.form['username']})

            if existing_user is None:
                #hashpass = bcrypt.hashpw(request.form['pass'].encode('utf-8'), bcrypt.gensalt())
                pw_hash = bcrypt.generate_password_hash(passwords).decode('utf-8')
                users.insert({'name': request.form['username'], 'password': pw_hash})
                session['username'] = request.form['username']
                return render_template('mainindex.html')

            return render_template('register.html', error='That username already exists!')


    return render_template('register.html')
if __name__ == '__main__':
    app.debug = True
    app.run()

@app.route('/process',methods=['GET', 'POST'])
def process():
    if 'username' in session:
        if request.method == 'POST':
            sentence = request.form['user_input']
            username = session['username']
            print(username)
            tag_result = request.form.get('tags')
            tage_value = str(tag_result)
            question = sentence
            #sentence = tokenize(sentence)

            sentence = tokenize(sentence.lower())
            stopsets = ['a', 'an', 'the', 'i', 'you', 'one', 'do', 'have', 'of', 'in', 'like', 'for', 'from', 'to',
                        'as', 'by', 'about', 'off', 'did', 'am', 'is',
                        'are', 'was', 'were', 'if', 'is', 'on', 'what', 'why', 'when', 'where', 'which', 'and', 'how',
                        'tell',
                        'me', 'my', 'must', 'can', 'could', 'would', 'that', 'or', 'anyone', 'any', 'many', 'there']
            stopX = [stem(w) for w in sentence if w not in stopsets]
            print("\n>>>[", stopX)

            #X = bag_of_words(sentence, all_words)
            X = bag_of_words(stopX, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)
            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        if tag_result == 'Select':
                            return render_template('index.html', user_input=question,
                                                   bot_response=random.choice(intent['responses']),
                                                   error='Please select any option from the dropdown')
                        else:
                            # f = open("file.txt", "a")
                            # f.write(tage_value + '\n' + question + '\n' + random.choice(intent['responses']) + '\n')
                            print(username)
                            print(tage_value)
                            print(question)
                            print(random.choice(intent['responses']))
                            chatTable = mongo.db.Chat
                            chatData = chatTable.find({'name': username})
                            all_questions = []
                            all_answers = []
                            if chatData:
                                for x in chatData:
                                    questions = x['details']['question']
                                    answers = x['details']['answer']
                                    all_questions.append(questions)
                                    all_answers.append(answers)

                                return render_template('review.html', message=session['username'], user_names=username,
                                                       tag_value=tage_value,
                                                       user_input=question,
                                                       bot_response=random.choice(intent['responses']),
                                                       all_history=zip(all_questions, all_answers))
                            else:
                                return render_template('review.html', message=session['username'], user_names=username,
                                                       tag_value=tage_value,
                                                       user_input=question,
                                                       bot_response=random.choice(intent['responses']),
                                                       all_history=zip(all_questions, all_answers))
                        # print(f"{bot_name}: {random.choice(intent['responses'])}")
            else:
                if tag_result == 'Select':
                    return render_template('index.html', user_input=question, bot_response="I don't understand",
                                           error='Please select any option the dropdown')
                else:
                    # f = open("file.txt", "a")
                    # f.write(tage_value + '\n' + question + '\n' + 'I donot understand\n')
                    chatTable = mongo.db.Chat
                    chatData = chatTable.find({'name': username})
                    all_questions = []
                    all_answers = []
                    if chatData:
                        for x in chatData:
                            questions = x['details']['question']
                            answers = x['details']['answer']
                            all_questions.append(questions)
                            all_answers.append(answers)
                        return render_template('review.html', message=session['username'], user_names=username,
                                               tag_value=tage_value,
                                               user_input=question,
                                               bot_response="I don't understand",
                                               all_history=zip(all_questions, all_answers))
                    else:
                        return render_template('review.html', message=session['username'], user_names=username,
                                               tag_value=tage_value,
                                               user_input=question,
                                               bot_response="I don't understand",
                                               all_history=zip(all_questions, all_answers))
    return render_template('mainindex.html')

@app.route('/review',methods=['GET', 'POST'])
def review():
    if 'username' in session:
        if request.method == 'POST':
            select = request.form.get('review')
            session['username'] = request.form['names_input']
            username = session['username']
            tags_input = request.form['tags_input']
            questions_input = request.form['questions_input']
            answers_input = request.form['answers_input']
            value = str(select)
            print(value)
            print(username)
            print(tags_input)
            print(questions_input)
            print(answers_input)
            if value == 'Select':
                return render_template('review.html', user_input=questions_input, bot_response=answers_input,
                                       error='Please select Yes or No from the dropdown')
            else:
                chat = mongo.db.Chat
                chat.save({"name": username,'details':{'tag': tags_input, 'question': questions_input, 'answer':answers_input, 'review':value}})
            chatTable = mongo.db.Chat
            chatData = chatTable.find({'name': username})
            all_questions = []
            all_answers = []
            if chatData:
                for x in chatData:
                    questions = x['details']['question']
                    answers = x['details']['answer']
                    all_questions.append(questions)
                    all_answers.append(answers)

                return render_template('index.html', message=session['username'],
                                       all_history=zip(all_questions, all_answers))
            else:
                return render_template('index.html', message=session['username'],
                                       all_history=zip(all_questions, all_answers))
    return render_template('mainindex.html')

'''bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")'''
import random
import json
from flask import Flask, render_template, url_for, request, session, redirect
from flask.ext.pymongo import PyMongo
import bcrypt
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

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

#make flask ready
from flask import Flask, render_template, request, url_for, flash, redirect, session

app = Flask(__name__)
app.secret_key = '1212ewwrwrwrwrwrw2222'
@app.route('/')
def index():
    return render_template('usercreate.html')

@app.route('/process',methods=['GET', 'POST'])
def process():
    bot_name = "Sam"
    sentence = request.form['user_input']
    #user_name = request.form['names_input']
    session['username'] = request.form['names_input']
    username = session['username']
    print(username)
    tag_result = request.form.get('tags')
    tage_value = str(tag_result)
    question = sentence
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
                if tag_result == 'Select':
                    return render_template('index.html', user_input=question, bot_response=random.choice(intent['responses']),
                                           error='Please select any option from the dropdown')
                else:
                    #f = open("file.txt", "a")
                    #f.write(tage_value + '\n' + question + '\n' + random.choice(intent['responses']) + '\n')
                    x = question
                    y = random.choice(intent['responses'])
                    global a, b, tagvalue
                    a = x
                    b = y
                    tagvalue = tage_value
                    print(username)
                    print(tage_value)
                    print(question)
                    print(random.choice(intent['responses']))
                    return render_template('review.html',user_names = username,tag_value = tage_value, user_input=question,
                                           bot_response=random.choice(intent['responses']))
                #print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        if tag_result == 'Select':
            return render_template('index.html', user_input=question, bot_response="I don't understand",
                                   error='Please select any option the dropdown')
        else:
            #f = open("file.txt", "a")
            #f.write(tage_value + '\n' + question + '\n' + 'I donot understand\n')
            x = question
            y = "I don't understand"
            a = x
            b = y
            tagvalue = tage_value
            return render_template('review.html',user_names = username,tag_value = tage_value, user_input=question, bot_response="I don't understand")

@app.route('/review',methods=['GET', 'POST'])
def review():
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
    if value=='Select':
        return render_template('review.html', user_input=questions_input, bot_response=answers_input, error ='Please select Yes or No from the dropdown')
    else:
        f = open("FilePath/"+username+".txt", "a")
        f.write(tags_input + '\n' + questions_input + '\n' + answers_input + '\n' + value+'\n\n')
    return render_template('index.html')

@app.route('/create',methods=['GET', 'POST'])
def create():
    sentence = request.form['user_create']
    name_value = str(sentence)
    if name_value=='':
        return render_template('usercreate.html', error ='Please enter your name')
    else:
        f = open("FilePath/"+name_value+".txt", "a")
        global names
        names = name_value
        return render_template('index.html',name_values = name_value)

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
import random
import json

import printjson as printjson
from flask import Flask, render_template, url_for, request, session, redirect
from flask_pymongo import PyMongo
import bcrypt
from flask_bcrypt import Bcrypt
import torch

import requests
from bs4 import BeautifulSoup

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
#app.config['MONGO_URI'] = 'mongodb+srv://Shafkat:shafkaait@cluster0.mhvly.mongodb.net/chatdata'

mongo = PyMongo(app)

@app.route('/')
def index():
    if 'username' in session:
        username = session['username']
        chatTable = mongo.db.Chat
        chatData = chatTable.find({'name': username})
        all_questions = []
        all_answers = []
        all_yes_reviews = []
        all_no_reviews = []
        if chatData:
            for x in chatData:
                questions = x['details']['question']
                answers = x['details']['answer']
                reviews = x['details']['review']
                all_questions.append(questions)
                all_answers.append(answers)
                if reviews == 'Yes':
                    all_yes_reviews.append(reviews)
                elif reviews == 'No':
                    all_no_reviews.append(reviews)

            all_questions.reverse()
            all_answers.reverse()

            yes_count = len(all_yes_reviews)
            no_count = len(all_no_reviews)
            efficiency = (int(yes_count)/(int(yes_count)+int(no_count)))*100
            efficiency = round(efficiency, 2)
        return render_template('index.html',message= session['username'],all_history=zip(all_questions, all_answers),yes_count=yes_count,no_count=no_count,efficiency=efficiency)
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
        all_yes_reviews = []
        all_no_reviews = []
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
                        reviews = x['details']['review']
                        all_questions.append(questions)
                        all_answers.append(answers)
                        if reviews == 'Yes':
                            all_yes_reviews.append(reviews)
                        elif reviews == 'No':
                            all_no_reviews.append(reviews)
                    all_questions.reverse()
                    all_answers.reverse()
                    yes_count = len(all_yes_reviews)
                    no_count = len(all_no_reviews)
                    efficiency = (int(yes_count) / (int(yes_count) + int(no_count))) * 100
                    efficiency = round(efficiency, 2)
                    return render_template('index.html', message=session['username'],
                                           all_history=zip(all_questions, all_answers),yes_count=yes_count,no_count=no_count,efficiency=efficiency)
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
            stopsets = ['a', 'an', 'the', 'i', 'you', 'one', 'do', 'of', 'in', 'like', 'for', 'from', 'to',
                        'as', 'by', 'about', 'off', 'did', 'am', 'is',
                        'are', 'was', 'were', 'if', 'is', 'on', 'what', 'when', 'where', 'which', 'and',
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
                            all_yes_reviews = []
                            all_no_reviews = []
                            if chatData:
                                for x in chatData:
                                    questions = x['details']['question']
                                    answers = x['details']['answer']
                                    reviews = x['details']['review']
                                    all_questions.append(questions)
                                    all_answers.append(answers)
                                    if reviews == 'Yes':
                                        all_yes_reviews.append(reviews)
                                    elif reviews == 'No':
                                        all_no_reviews.append(reviews)
                                all_questions.reverse()
                                all_answers.reverse()
                                yes_count = len(all_yes_reviews)
                                no_count = len(all_no_reviews)
                                efficiency = (int(yes_count) / (int(yes_count) + int(no_count))) * 100
                                efficiency = round(efficiency, 2)

                                return render_template('review.html', message=session['username'], user_names=username,
                                                       tag_value=tage_value,
                                                       user_input=question,
                                                       bot_response=random.choice(intent['responses']),
                                                       all_history=zip(all_questions, all_answers),yes_count=yes_count,
                                                       no_count=no_count,efficiency=efficiency)
                            else:
                                return render_template('review.html', message=session['username'], user_names=username,
                                                       tag_value=tage_value,
                                                       user_input=question,
                                                       bot_response=random.choice(intent['responses']),
                                                       all_history=zip(all_questions, all_answers))
                        # print(f"{bot_name}: {random.choice(intent['responses'])}")
            else:
                print("hello")
                if tag_result == 'Select':
                    return render_template('index.html', user_input=question, bot_response="I don't understand",
                                           error='Please select any option the dropdown')
                else:
                    # f = open("file.txt", "a")
                    # f.write(tage_value + '\n' + question + '\n' + 'I donot understand\n')
                    bot_response= web_scraping(question)
                    chatTable = mongo.db.Chat
                    chatData = chatTable.find({'name': username})
                    all_questions = []
                    all_answers = []
                    all_yes_reviews = []
                    all_no_reviews = []
                    if chatData:
                        for x in chatData:
                            questions = x['details']['question']
                            answers = x['details']['answer']
                            reviews = x['details']['review']
                            all_questions.append(questions)
                            all_answers.append(answers)
                            if reviews == 'Yes':
                                all_yes_reviews.append(reviews)
                            elif reviews == 'No':
                                all_no_reviews.append(reviews)
                        all_questions.reverse()
                        all_answers.reverse()
                        yes_count = len(all_yes_reviews)
                        no_count = len(all_no_reviews)
                        efficiency = (int(yes_count) / (int(yes_count) + int(no_count))) * 100
                        efficiency = round(efficiency, 2)

                        return render_template('review.html', message=session['username'], user_names=username,
                                               tag_value=tage_value,
                                               user_input=question,
                                               bot_response=bot_response,
                                               all_history=zip(all_questions, all_answers),yes_count=yes_count,no_count=no_count,efficiency=efficiency)
                    else:
                        return render_template('review.html', message=session['username'], user_names=username,
                                               tag_value=tage_value,
                                               user_input=question,
                                               bot_response=bot_response,
                                               all_history=zip(all_questions, all_answers))
    return render_template('mainindex.html')


def web_scraping(qs):
    global flag2
    global loading

    URL = 'https://www.google.com/search?q=' + qs
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, 'html.parser')

    links = soup.findAll("a")
    all_links = []
    for link in links:
        link_href = link.get('href')
        if "url?q=" in link_href and not "webcache" in link_href:
            all_links.append((link.get('href').split("?q=")[1].split("&sa=U")[0]))

    flag = False
    for link in all_links:
        if 'https://en.wikipedia.org/wiki/' in link:
            wiki = link
            flag = True
            break

    div0 = soup.find_all('div', class_="kvKEAb")
    div1 = soup.find_all("div", class_="Ap5OSd")
    div2 = soup.find_all("div", class_="nGphre")
    div3 = soup.find_all("div", class_="BNeawe iBp4i AP7Wnd")

    if len(div0) != 0:
        answer = div0[0].text
    elif len(div1) != 0:
        answer = div1[0].text + "\n" + div1[0].find_next_sibling("div").text
    elif len(div2) != 0:
        answer = div2[0].find_next("span").text + "\n" + div2[0].find_next("div", class_="kCrYT").text
    elif len(div3) != 0:
        answer = div3[1].text
    elif flag == True:
        page2 = requests.get(wiki)
        soup = BeautifulSoup(page2.text, 'html.parser')
        title = soup.select("#firstHeading")[0].text

        paragraphs = soup.select("p")
        for para in paragraphs:
            if bool(para.text.strip()):
                answer = title + "\n" + para.text
                break
    else:
        answer = "Sorry. I could not find the desired results"


    flag2 = False

    return answer


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
            all_yes_reviews = []
            all_no_reviews = []
            if chatData:
                for x in chatData:
                    questions = x['details']['question']
                    answers = x['details']['answer']
                    reviews = x['details']['review']
                    all_questions.append(questions)
                    all_answers.append(answers)
                    if reviews == 'Yes':
                        all_yes_reviews.append(reviews)
                    elif reviews == 'No':
                        all_no_reviews.append(reviews)
                all_questions.reverse()
                all_answers.reverse()
                yes_count = len(all_yes_reviews)
                no_count = len(all_no_reviews)
                efficiency = (int(yes_count) / (int(yes_count) + int(no_count))) * 100
                efficiency = round(efficiency, 2)

                return render_template('index.html', message=session['username'],
                                       all_history=zip(all_questions, all_answers),yes_count=yes_count,no_count=no_count,efficiency=efficiency)
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
from flask import Flask, url_for, render_template, send_file
from werkzeug.utils import secure_filename
from flask import request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from service import blogopen
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///loginDB.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


@app.errorhandler(404)
def page_not_found(error):
    app.logger.error(error)
    return render_template('page_not_found.html'), 404


class User(db.Model):
    """Create user table"""
    __table_name__ = 'user'


    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password = db.Column(db.String(80))
    email = db.Column(db.String(80), unique=True)
    age = db.Column(db.Integer)
    sex = db.Column(db.String(80))
    act = db.Column(db.Integer)#근데 사실 활동량을 뭐라고 적어야할까 사용자 입장에선.. 난감할듯

    def __init__(self, username, password, email):
        self.username = username
        self.password = password
        self.email = email


@app.route('/homepage', methods=['GET', 'POST'])
def home():
    if not session.get('logged_in'):
        return render_template('login/index.html')
    else:
        if request.method == 'POST':
            username = request.form['username']
            return render_template('login/index.html', data=blogopen(username))
        return render_template('login/index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login form"""
    if request.method == 'POST':
        name = request.form['username']
        passw = request.form['password']
        try:
            data = User.query.filter_by(username=name, password=passw).first()
            if data is not None:
                session['logged_in'] = True
                return redirect(url_for('home'))
            else:
                return '여기가 안먹음'
        except:
            return "그럼 여기?"

    else:
            return render_template('login/login.html')

@app.route('/register/', methods=['GET', 'POST'])
def register():
    """Register Form"""
    if request.method == 'POST':
        new_user = User(username=request.form['username'], password=request.form['password'],
                        email=request.form['email'])
        db.session.add(new_user)
        db.session.commit()
        return render_template('login/login.html')
    return render_template('login/register.html')


@app.route('/logout')
def logout():
    """Logout Form"""
    session['logged_in'] = False
    return redirect(url_for('home'))


@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('./uploads/' + secure_filename(f.filename))
        return render_template('check.html')
    else:
        return render_template('page_not_found.html')


@app.route('/downfile')
def down_page():
    files = os.listdir("./uploads")
    return render_template('filedown.html', files=files)


@app.route('/fileDown', methods=['GET', 'POST'])
def down_file():
    if request.method == 'POST':
        sw = 0
        files = os.listdir("./uploads")
        for x in files:
            if (x == request.form['file']):
                sw = 1
                path = "./uploads/"
                return send_file(path + request.form['file'],
                                 attachment_filename=request.form['file'],
                                 as_attachment=True)
        return render_template('page_not_found.html')
    else:
        return render_template('page_not_found.html')


if __name__ == '__main__':
    db.create_all()
    app.run()

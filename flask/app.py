import json
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, url_for, render_template, send_file, make_response
from werkzeug.utils import secure_filename
from flask import request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from service import blogopen
from PIL import ImageDraw, Image, ImageFont
import sqlite3

import os

from models import *
from utils.datasets import *
from utils.utils import *

import pandas as pd
import datetime


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///loginDB.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

fontpath = 'data/NanumBarunGothic.ttf'
font = ImageFont.truetype(fontpath, 10)


class Name():
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

# 사용자 이름 변수
name = Name("None")


# default error 화면
@app.errorhandler(404)
def page_not_found(error):
    app.logger.error(error)
    return render_template('page_not_found.html'), 404

# user table 생성 -> 회원가입시 사용자 정보 등록
class User(db.Model):
    """Create user table"""
    __table_name__ = 'user'

    username = db.Column(db.String(80), primary_key=True)
    password = db.Column(db.String(80))
    email = db.Column(db.String(50), unique=True)
    age = db.Column(db.String(80))
    sex = db.Column(db.String(80))
    act = db.Column(db.String(80))

    def __init__(self, username, password, email, age, sex,act):
        self.username = username
        self.password = password
        self.email = email
        self.age = age
        self.sex = sex
        self.act = act

# 영양소 table 생성  -> 크롤링 음식 영양소 정보
class User_food(db.Model):
    """Create user table"""
    __table_name__ = 'user_food'
    column_not_exist_in_db = db.Column(db.Integer, primary_key=True)
    username = db.Column('username', db.String(80), db.ForeignKey('user.username'))
    date = db.Column('날짜', db.DateTime, default=datetime.datetime.now())
    food = db.Column('FOOD_LABEL', db.String(80))
    a1 = db.Column('단백질', db.Float, nullable=False)
    a2 = db.Column('탄수화물', db.Float, nullable=False)
    a3 = db.Column('지방', db.Float, nullable=False)
    a4 = db.Column('철분', db.Float, nullable=False)
    a5 = db.Column('오메가3', db.Float, nullable=False)
    a6 = db.Column('칼슘', db.Float, nullable=False)
    a7 = db.Column('비타민D', db.Float, nullable=False)
    a8 = db.Column('아연', db.Float, nullable=False)
    a9 = db.Column('비타민B12', db.Float, nullable=False)

    def __init__(self, username, date, food, a1,a2,a3,a4,a5,a6,a7,a8,a9):
        self.username = username
        self.date = date
        self.food = food
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6
        self.a7 = a7
        self.a8 = a8
        self.a9 = a9


# @app.route('/homepage', methods=['GET', 'POST'])
# def home():
#     if not session.get('logged_in'):
#         return render_template('login/index.html')
#     else:
#         if request.method == 'POST':
#             username = request.form['username']
#             return render_template('login/index.html', data=blogopen(username))
#         return render_template('login/index.html')

#로그인 페이지 -> 사용자의 이름과 비밀번호를 통해 로그인 성공시 main homepage로 이동.
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login form"""
    if request.method == 'POST':
        global name
        name = request.form['username']
        passw = request.form['password']
        try:
            data = User.query.filter_by(username=name, password=passw).first()
            if data is not None:
                session['logged_in'] = True
                # session['username'] = name
                return redirect(url_for('home'))
            else:
                return 'hi'
        except:
            return render_template('index.html', user=request.form['username'])  # 로그인 성공시 사용자의 username 가져오기.

    else:
        return render_template('login/login2.html')

# 회원가입 페이지 -> 회원가입할때 입력한 정보를 user table 에 등록.
@app.route('/register/', methods=['GET', 'POST'])
def register():
    """Register Form"""

    if request.method == 'POST':
        new_user = User(username=request.form['username'], password=request.form['password'],
                        email=request.form['email'], age=request.form['age'],
                        sex=request.form['sex'], act=request.form['act'])
        db.session.add(new_user)
        db.session.commit()
        return render_template('login/login2.html')
    return render_template('login/register2.html')

# 로그아웃 페이지
@app.route('/logout')
def logout():
    """Logout Form"""
    session['logged_in'] = False
    print("hi")
    session.pop('username', None)
    return redirect(url_for('home'))


@app.route('/')
def home_page():
    return render_template('login/register2.html')

# 파일 업로드를 위한 페이지
@app.route('/upload')
def upload_page():
    session['username'] = session.get('username')
    return render_template('upload.html')

# 파일 업로드를 처리하는 페이지 -> 아침, 점심, 저녁 식단에 해당하는 음식 이미지를 업로드
#                           -> 업로드한 이미지는 학습시킨 모델에 적용
#                           -> object detection 을 통해 나온 인식된 객체 이름에 해당하는 영양소 정보를 user_food 와 매치시켜 영양소 정보를 가져옴
#                           -> 섭취한 영양소별 그룹화를 통해 섭취량 계산
#                           -> 사용자의 나이, 성별에 따른 권장섭취량과 비교
#                           -> 제일 부족한 영양소 2개를 추출
#                           -> 자체 개발한 추천 시스템을 통해 메뉴 5가지를 추천
@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        global name
        f = request.files['file']
        f.save('./uploads/' + secure_filename(f.filename))

        a = yolo_predict('./uploads/' + secure_filename(f.filename))

        sunyoung = secure_filename(f.filename)
        print("sunyoung:{}".format(sunyoung))
        con = sqlite3.connect('food_label.db')
        cur = con.cursor()
        a = list(set(a))
        food = [_.strip() for _ in a]

        for i in food:
            cur.execute("SELECT * FROM food_label where food_label.FOOD_LABEL=='{}'".format(i))
        rows = cur.fetchall()
        cur.close()
        con.close()

        for row in rows:
            new_food = User_food(username=name, date=datetime.datetime.now(),
                                 food=row[0], a1=row[1],a2=row[2],a3=row[3],a4=row[4],
                                 a5=row[5],a6=row[6],a7=row[7],a8=row[8],a9=row[9],)
            db.session.add(new_food)
            db.session.commit()

        user = User.query.filter_by(username=name).first()
        s = user.sex
        a = user.age

        con2 = sqlite3.connect('promote.db')
        cur2 = con2.cursor()
        cur2.execute("SELECT * FROM standard where standard.sex=='{}' and standard.age == '{}'".format(s, a))
        stand = cur2.fetchone()[3:]
        print(stand)
        cur2.close()
        con2.close()

        now = datetime.datetime.now()
        yesterday = (now - datetime.timedelta(1)).date()

        # 시간 데이터를 통해 아침, 점심, 저녁에 등록한 식단을 구분하고 섭취 영양소를 계산, 부족한 영양소를 추출
        in_take = User_food.query.filter_by(username=name)
        df = pd.read_sql(in_take.statement, in_take.session.bind)
        df['날짜'] = pd.to_datetime(df['날짜'], format='%Y-%m-%d %H:%M:%S', errors='raise')
        yesterday = datetime.datetime.now() - datetime.timedelta(1)
        df_leak = df[df['날짜'].dt.date == yesterday.date()].loc[:, 'FOOD_LABEL':'비타민B12']


        leak = pd.Series(stand, index=df_leak.columns[1:]) - df_leak.iloc[:, 1:].sum()
        leak_nor = leak / pd.Series(stand, index=df_leak.columns[1:])
        leak_nutri = leak_nor.sort_values(ascending=False)[:2]
        print(leak_nutri)

        with open('data/necc_nutri.json', "r") as json_file:
            necc_nutri = json.load(json_file)


        recomm = []
        for i in leak_nutri.index:
            for j in necc_nutri[i]:
                if j in recomm:
                    pass
                else:
                    recomm.append(j)
        print(recomm)
        random_recomm = random.sample(recomm,5)
        print(random_recomm)

        # 선정된 5가지 메뉴 이름을 변수로 저장하여 first.html(main page)로 값 보내기.
        sfile1 = '../static/images/food/' + random_recomm[0] + '.jpg'
        sfile2 = '../static/images/food/' + random_recomm[1] + '.jpg'
        sfile3 = '../static/images/food/' + random_recomm[2] + '.jpg'
        sfile4 = '../static/images/food/' + random_recomm[3] + '.jpg'
        sfile5 = '../static/images/food/' + random_recomm[4] + '.jpg'


        return render_template('first.html', user_naming= user, val_textt=leak_nutri, sy1 = sfile1, sy2 = sfile2, sy3 = sfile3, sy4 = sfile4, sy5 = sfile5, name1 = random_recomm[0], name2 = random_recomm[1],name3 = random_recomm[2],name4 = random_recomm[3], name5 = random_recomm[4])

        # return render_template('visualize.html', sy = sunyoung, one_i = random_recomm[0], two_i = random_recomm[1], three_i= random_recomm[2], four_i = random_recomm[3], five_i = random_recomm[4])
    else:
        return render_template('page_not_found.html')

@app.route('/index/first')
def first():
    return render_template('first.html')

@app.route('/downfile')
def down_page():
    files = os.listdir("./uploads")
    return render_template('filedown.html', files=files)


@app.route('/live-data')
def live_data():
    # Create a PHP array and echo it as JSON
    data = [time() * 1000, random() * 100]
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response


@app.route("/graph")
def graph():
    return render_template('graph.html')

# 올린 이미지 식단에 대한 목록들을 볼 수 있는 페이지.
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

# 이미지를 업로드하면 학습한 모델 값에 적용
@app.route('/predict', methods=['GET', 'POST'])
def yolo_predict(image_name = None):

    cfg = check_file('data/yolov3-spp31.cfg')
    names = check_file('data/classes.names')
    imgsz = 512
    a=[]
    with torch.no_grad():
        out, source, weights, view_img, save_txt, save_img, augment = './output/', image_name, 'data/best.pt', False, False, True, True
        device = torch_utils.select_device(device='cpu')
        print(check_file(source))

        global model
        model = Darknet(cfg, imgsz)
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        model.to(device).eval()

        vid_path, vid_writer = None, None
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

        names = load_classes(names)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.float()) if device.type != 'cpu' else None  # run once

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=augment)[0]
            t2 = torch_utils.time_synchronized()
            pred = non_max_suppression(pred, 0.15, 0.6, multi_label=False, classes=None, agnostic=False)

            for i, det in enumerate(pred):  # detections for image i

                p, s, im0 = path, '', im0s
                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from imgsz to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                                file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format


                        label = '%s %.2f' % (names[int(cls)], conf)
                        a.append(names[int(cls)])
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                            # output label
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        img_pil = Image.fromarray(im0)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text((c1[0], c1[1] - 10), label, font=font, fill=(255, 255, 255))
                        im0 = np.array(img_pil)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
    return a

#결과값 페이지
@app.route("/result")
def result():
    return render_template("result.html")

@app.route("/result_data", methods=['GET','POST'])
def result_data():
    if request.method=='GET':
        result_name = request.form['result_name']
        return "{}님 환영합니다.".format(result_name)
    else:
        result_name = request.form['result_name']
        return "{}님 환영합니다.".format(result_name)

@app.route("/index")
def index_home():
    return render_template("index.html")

if __name__ == '__main__':
    db.create_all()
    app.secret_key = 'super secret key'
    app.run(debug=True)

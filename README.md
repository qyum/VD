How to run the server: This is a fastapi server that takes in request from the app and sends back the request Follow these steps to run in Windows(VScode)

    1.download & install python version==3.7,tensorflow ==1.13.0 and keras == 2.2.4

    2.git clone https://github.com/qyum/VD.git

    3.Create a virtual environment & activate it in python.

        python -m venv myenv  
        .\myenv\Scripts\activate  
        

    4.cd VD

    5.use pip install -r requirements.txt
    6.Run python -m uvicorn main:app --reload
    8.The server should now be running at http://127.0.0.1:8000/health

    9.Api docs are available at http://127.0.0.1:8000/docs

    10.You can make requests at the endpoint http://127.0.0.1:8000/detect


Necessary Files:
    1.Weights: I've attached a best weight of training model.please check it out. https://drive.google.com/file/d/1yF9kTYzfeugPJsqGQQem53ocU42g5Bhs/view?usp=sharing

2.Loss graph:
    https://drive.google.com/file/d/14-r-H5z5bfWGBepbGWXt83pImF5dnJXt/view?usp=sharing

DemoVideo: https://drive.google.com/file/d/1p0ZdN4BYM_Y79_pijeAnyrC7MBUaixjV/view?usp=sharing

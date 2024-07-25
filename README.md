<div align="center">

# BING: Action-Recognition

*BING 서비스 아키텍쳐에 사용된 행동인식 모델 엔드포인트 코드 입니다.*

[![Static Badge](https://img.shields.io/badge/language-english-red)](./README.md) [![Static Badge](https://img.shields.io/badge/language-korean-blue)](./README-KR.md) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FSinging-voice-conversion%2Fsingtome-model&count_bg=%23E3E30F&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>

<br>

**SW중심대학 디지털 경진대회** : SW와 생성 AI의 만남 - SW 부문
팀 GASBY의 BING 서비스
이 리포지토리는 팀 GASBY가 SW중심대학 디지털 경진대회에서 개발한 BING 서비스에 사용된 AWS Lambda 함수의 코드를 포함하고 있습니다. 본 프로젝트는 생성 AI 기술을 활용하여 사용자의 요구에 맞는 다양한 소프트웨어 솔루션을 제공합니다.

**레포지토리 개요**: 
BING 서비스는 최신 생성 AI 알고리즘을 사용하여 실시간으로 데이터를 처리하고 사용자에게 맞춤형 결과를 제공합니다. 이 프로젝트는 서버리스 아키텍처를 기반으로 하며, AWS Lambda를 핵심 컴퓨팅 리소스로 사용합니다.

**주요 기능**: 

실시간 데이터 처리: 사용자의 요청을 실시간으로 처리하여 빠르고 정확한 결과를 제공합니다.

생성 AI 통합: 최신 AI 모델을 활용하여 사용자 요구에 맞춤형 결과 생성.

서버리스 아키텍처: AWS Lambda를 사용하여 확장성과 비용 효율성을 극대화.

<br>

<div align="center">

<h3> SERVICE part Team members </h3>

| Profile | Name | Role |
| :---: | :---: | :---: |
| <a href="https://github.com/wooing1084"><img src="https://avatars.githubusercontent.com/u/32007781?v=4" height="120px"></a> | SungHoon Jung <br> **wooing1084**| Pytorch R2+1D Model Training, <br> Predictive Endpoint Development |

<br>


</div>

<br>

## 1. 레포지토리 소개
학습된 R2+1D모델을 사용하여 행동인식 예측하는 서버 코드가 포함되어있습니다.

### 기술 스택
flask - 3.0.3

pytorch - 2.3.1

### 파일 정보
app.py - flask 서버 메인 파일입니다.

player.py - 선수 정보를 담는 객체입니다. 팀, 바운딩박스 리스트, 위치, 행동등을 담고있습니다.

action_recognition.py - R2+1D모델로 행동인식하는 로직이 구현되어있는 코드입니다.

checkpoints.py - 학습된 pt파일로 행동인식 모델을 불러오는 로직이 구현되어있는 코드입니다.

s3utils.py - S3에 업로드, 다운로드를 위한 유틸리티 파일입니다.

## 2. 실행 방법
1. model_checkpoints/r2plus1d_augemnted-2 폴더에 학습된 r2plus1d_multiclass_19_0.0001.pt파일을 붙여넣습니다.
2. .env파일을 생성하고 아래와같이 작성합니다.
```
AWS_Accesskey= 제공한 AWS 엑세스 키
AWS_Secretkey= 제공한 AWS 비밀 키
AWS_Region= 제공한 AWS 리전 정보
```
3. 터미널에 python3 app.py 입력하여 app.py를 실행한다.




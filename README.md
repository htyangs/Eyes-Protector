## Eye-Protectors  [(Homepage)](https://www.htyang.com/projects)

## Feature
###  ⇾ ⇾ 完整中文介紹[請點這邊](https://sites.google.com/view/eyeprotector/%E9%A6%96%E9%A0%81) ⇽ ⇽
<p align="center">
<img src="https://user-images.githubusercontent.com/29053630/174329360-557c568a-6999-4890-b33b-d7ef0ba37d50.png" width="1000">
<p/> 

## How to install
### Windows : download the Eye protector.exe file [here](https://docs.google.com/uc?id=12r2WWWyhLgpVB1fcmofMiVNxxdPcjXPL&confirm=t) and execute it. 
1. Since the program is built by Pyinstaller and does not certificate by an official organization, the windows defender will be alert about it. Just click "trust the program" and give permission to it! [Reference](https://stackoverflow.com/questions/54733909/windows-defender-alert-users-from-my-pyinstaller-exe)
2. If you install the program under C:\Program Files, please right click the eye.exe file and run as administrator. 
### MacOS : download the Eye protector.exe file <a href="https://docs.google.com/uc?id=1norVbEBE5NiB0g6xhuDt7naZBAZv4uA1&confirm=t" target="_blank">here</a> and execute it. 
1. Type "sudo spctl –master-disable" in the terminal and choose "allow apps downloaded from anywhere" in the Security & Privacy panel. [tutorial](https://www.macworld.com/article/672947/how-to-open-a-mac-app-from-an-unidentified-developer.html) 
2. Make the file executable by typing **chmod 755 "path to the file"** in the terminal. EX: chmod 755 /Users/alwaysmle/Downloads/eye. 
3. Simply execute the program after changing the type of program.
### Run Python File Directly
1. pip install -r requirements.txt
2. python eye.py
## How to use?
### Main program
<p align="center">
<img src="https://user-images.githubusercontent.com/29053630/173589305-13fdd1ff-d248-4ad7-8ebf-6988bcf8caff.png" width="1000">
<p/> 

1. **Add user**: type your name (do not use numbers or symbols). The program will save the name, threshold, and daily usage of eyes in the database of this user. This step can be ignored if you had added the user previously. The default user is "None" if one does not want to add any user.
2. **Choose user**: choose the user you want to save the data to.
3. **Confirm**: the program will read the threshold and line token(if one had saved preciously) from the database
4. **Use Line(Optional)** :
check the box if you want the program to send the notification to Line as a message. Line token for individual users is needed ([Line token 申請教學](https://xenby.com/b/274-%E6%95%99%E5%AD%B8-%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8-line-notify-%E5%B0%8D%E7%BE%A4%E7%B5%84%E9%80%B2%E8%A1%8C%E9%80%9A%E7%9F%A5) ,[Line notification api](https://notify-bot.line.me/doc/en/))
5. **Initialize**: the program detects the eye aspect ratio, relative distance, and relative brightness of the environment. As a result, one should set the standard distance, brightness, and eye aspect ratio so that the program can find a reasonable threshold for each user. 
6. **Modify the threshold**: the user can still modify the threshold to an appropriate value after initialization.
7. **Set the working time and resting time.**
8. **Exercise**: Select the exercise you want to do during the resting time.
9. **Start**: The program will start to count down the working time.

### Analysis
<p align="center">
<img src="https://user-images.githubusercontent.com/29053630/173602693-7034327e-d075-4df6-8f14-21496ff8e802.png" width="1000">
<p/> 

1. **Choose the user**
2. **Choose the date**
3. The user's state, environmental brightness, blink frequency, and eye distance will be shown in the right channel. In the line chart of the user's state, 2 stands for working time, 1 stands for "not detect people", and 0 stands for resting time.

## Demo (click the image and transfer to youbute)

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/29053630/173597982-7cdca861-dc04-45f8-a1be-a5f5b4587f10.png)](https://www.youtube.com/watch?v=uCtQtaZsr3g)

## Donation

### It has taken me a long time to develop this program. Your kind donation of a cup of coffee  will go a long way in helping me improve the program. [Donation Link](https://www.buymeacoffee.com/eye_protector) is here, thanks!


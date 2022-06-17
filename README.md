## Eye-Protectors
### help the computer users protect their eyes from eyestrain, dry eyes and myopia.

## Feature
* Detect blinking frequency, eye distance between screen and environment brightness <br>
* Notify the users immediately by messagebox or Line.
* Remind the user to relax every certain minutes.
* Check the user relax their eyes enough by 1.dectecting the closing time of eyes 2.excerise (jumping jack). 
* Provide daily analysis report for the usage of eye.
* Working times will only be counted if the program dectect the users 

## How to use?
### Main program
1. **Add user** : type your name (do not use numbers or symbols). The program will save the name, threshold, daily usage of eyes in the database of this user. This step can be ignored if you had added the user previously.
2. **Choose user** : choose the user you want to save the data to.
3. **Confirm :** the program will read the threshold and line token(if one had saved preciously) from the database
4. **Use Line(Optional) :** <br>check the box if you want the program to send the notification to Line as a message. Line token for individual users is needed ([Line token 申請教學](https://jackterrylau.pixnet.net/blog/post/228035426-2019-08-09%E7%94%B3%E8%AB%8B%E4%B8%80%E5%80%8Bline-notify-token-%E4%BE%86-%E7%94%A8line-%E5%B9%AB%E4%BD%A0) ,[Line notification api](https://notify-bot.line.me/doc/en/))
5. **Initialize :** the program detect the eye aspect ratio, relative distance and relative brightness of environment. As a result, one should set the standard distance, brightness and eye aspect ratio so that the program can find a reasonable threshold for each users.
6.**Modify the threshold :** the user can still modify the threshold to an appropriate value after intialization.
7. **Set the working time and resting time.**
8. **Exercise :** Select the excercise you want to do during the resting time.
9. **Start :** The program will start to count down the working time.

<p align="center">
<img src="https://user-images.githubusercontent.com/29053630/173589305-13fdd1ff-d248-4ad7-8ebf-6988bcf8caff.png" width="800">
<p/> 

### Analysis
1. **Choose the user**
2. **Choose the date**
3. User's state, environmental brightness, blink freequency, eye distance will be showed in the right channel. In the line chart of user's state, 2 stands for working time, 1 stands for "not detect people", and 0 stands for resting time.
<p align="center">
<img src="https://user-images.githubusercontent.com/29053630/173602693-7034327e-d075-4df6-8f14-21496ff8e802.png" width="800">
<p/> 

## Demo (click the image and transfer to youbute)

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/29053630/173597982-7cdca861-dc04-45f8-a1be-a5f5b4587f10.png)](https://www.youtube.com/watch?v=uCtQtaZsr3g)

## How to install
### Windows : download the Eye protector.exe file and execute it. 
##### Since the program is built by Pyinstaller and does not certicate by official organization, the windows defender will alert about it. Just trust the program and give the permission to it! [Reference](https://stackoverflow.com/questions/54733909/windows-defender-alert-users-from-my-pyinstaller-exe)
### MacOS

## How it works?
1. **Blink eyes** : use mediapipe to mark the landmark around eyes and calculate the eye aspect ratio as below.
2. **Distance between eyes and screen** : the distance between eyes and screen is inversely proportional to the the area of eyes.
3. **Jumping jack** : use mediapipe to mark the skeleton of people and calculate the displacement of shoulder and hand.

<table align="center">
  <tr >
    <td style="text-align: center; vertical-align: middle;"><img src="https://user-images.githubusercontent.com/29053630/173599775-0d76e8df-e5ce-4d05-b66e-5b07bfaf8972.png" width="300"/>
    <td style="text-align: center; vertical-align: middle;"><img src="https://user-images.githubusercontent.com/29053630/173600462-086325e1-2dce-4e43-9e35-3d028b0d12a8.png" width="300"/>
  </tr>
 </table>

#####################
# ABOUT THE PROJECT #
#####################

Introduction:
The aim is to end up with a pipeline where it is easy to add a video of a badminton match, and then it will be analysed to provide the players with valuable insight about how to improve the game. 

What kind of analytics will it look at:

- Heatmap 
	- Individual player to see where the player most dominate on the court
	- Overlay af Heatmap to visualize how the players has moved to collaborate
	- Live visualizing in 2D to see a live view of how the players move
	- Live analytics of when the players move in a less desired way to compliment each other (I think I will need to track the shuttle too)

- Body Pose
	- does the body pose indicate that the player is not ready
	- does the body pose indicate that the player is streaching to reach the ball

####################################
# INITIAL PROCESS [FIRST TIME USE] #
####################################	

If there is no .venv folder in the repo on you pc, then you need to install one using:

python -m venv .venv

If there already is one available, then go to the root of the project and use the below to activate it

.\.venv\Scripts\Activate.ps1


########################
# Install Requirements #
########################

winget install Python.Python.3.11
deactivate
Remove-Item .venv -Recurse -Force
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

pip install -r requirements.txt
# uu-methods-ai-research
This repo contains the assignments made for the Methods in A.I. Research Course, Utrecht University, 1st Period.

## Usage
The user is prompted initially with a multi-select of the available configuration options. Choose between them with the arrow keys, toggle an option with SPACEBAR and finalize seletion with ENTER. After that the user has to select one of the 4 available dialog classifiers (1-4). Default is keyword matching. The bot is then available to chat with.

## Build Dependencies
This project uses [pyenv](https://github.com/pyenv/pyenv) in conjunction with [poetry](https://python-poetry.org/) for better package management. 
Assuming both are **properly installed** and you're **inside the directory** where the repository has been cloned, run `poetry install` and `poetry shell` to install all dependencies and setup a local virtualenv with said packages, respectively. `poetry run main.py` will run the main file inside the virtualenv. Having said that, poetry's and pyenv's usage is not stricly necessary, if you have a python environment with all packages already installed, feel free to just run `python main.py <args>`

## Files
- `main.py` - The file you're actually supposed to run. Contains the logic for building the models and starting the bot up.
- `bot.py` - Contains most of the bot logic and the dialog state system.
 - `restaurant.py` - Contains dataset and functions that get and set restaurant recommendations based on user preferences.
- `type_match_ls.py` - Contains logic for finding and setting all extra preferences, and returns that information to the bot.

## Arguments
These are the supported arguments when running `main.py` (i.e. `poetry run python main.py remodel`): 
1. `reprocess` -- forces the program to rebuild and reprocess the dataframe from the original dataset (dialog_acts.dat)
2. `remodel` -- forces the remodelling of the current models selected.
3. `debug` -- runs the whole program in debug mode, this enables LOTS of extra prints, so don't use this unless you actually plan on debugging something.

## Example Dialog Snippets
### Example 1
```
system: Hello! What kind of restaurant are you looking for?
user: I am looking for cheap italian food in the south. [you can specify any number of preferences]
s: Do you have any extra preferences?
u: no
s: You want to eat in a restaurant that has italian cuisine somewhere around the south in the cheap price range, did I get that right?
u: no
s: Hello! What kind of restaurant are you looking for? [back to the beginning]
u: bye
s: Goodbye! Thanks for using our bot!
```

### Example 2
```
s: Hello! What kind of restaurant are you looking for?
u: I am looking for cheap italian food.
s: Where would you like to eat?
u: In the south of the city.
s: Do you have any extra preferences?
u: no
s: You want to eat in a restaurant that has italian cuisine somewhere around the south in the cheap price range, did I get that right?
u: yes
s: [suggestion]...Is that ok?
u: no
s: Found 0 matching restaurant(s).
s: Sorry, there are no restaurants that conforms to your requirements.
s: Goodbye! Thanks for using our bot!
```

### Example 3
```
s: Hello! What kind of restaurant are you looking for?
u: chinese
s: Where would you like to eat?
u: any
s: How pricy you want the food to be?
u: any
s: Do you have any extra preference?
u: I want it to be romantic
s: You want to eat in a restaurant that has chinese cuisine, did I get that right?
u: yes
s: Found 2 matching restaurant(s).
s: [suggestion1] [romantic reasoning]
s: Is that ok?
u: no
s: Found 1 matching restaurant(s).
s: [suggestion2] [romantic reasoning]
s: Is that ok?
u: no
s: Found 0 matching restaurant(s).
s: Sorry, there are no restaurants that conforms to your requirements.
s: Goodbye! Thanks for using our bot!
```

#### Example 4
```
s: Hello! What kind of restaurant are you looking for?
u: Looking for an Italian restaurant.
s: Where would you like to eat?
u: In the east part of town.
s: How pricey you want the food to be?
u: moderate
s: Do you have any extra preference?
u: no thanks
s: You want to eat in a restaurant that has italian cuisine somewhere around the east in the moderate price range, did I get that right?
u: yes
s: Found 1 matching restaurant(s).
s: [suggestion1]
s: Is that ok?
u: yes
s: Goodbye! Thanks for using our bot!
```

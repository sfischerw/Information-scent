from flask import Flask
from flask import render_template
import json

app = Flask(__name__)

@app.route('/')
def hello_world():

    mon_dico = {'feature': 'A/C'}
    mon_titre = "Infoscent"

    gui = { "name": "Home", 
            "children": [ { "name": "Play radio", "size": 39}, 
                          { "name": "Car", 
                            "children": [ { "name": "Air conditioning", "size": 43}, 
                                          { "name": "Driving assistance", 
                                            "children": [ {"name": "Lane change alert", "size": 938}, 
                                                          {"name": "GPS", "size": 743},
                                                          { "name": "Cruise control", 
                                                            "children": [ {"name": "Activate cruise control", "size": 9938}, 
                                                                          {"name": "Turn off cruise control", "size": 9743} 
                                                                        ] },
                                                          {"name": "Anti-theft notification", "size": 743} 
                                                        ] },
                                        ] 
                          },
                         { "name": "Phone", 
                            "children": [ { "name": "Dial number", "size": 74}, 
                                          { "name": "Contact list", "size": 74}, 
                                          { "name": "Check voicemail messages", "size": 74}
                                        ]
                         }
                        ]
          } 



    return render_template('index.html', title=mon_titre, gui=json.dumps(gui))

if __name__ == '__main__':
    app.run()



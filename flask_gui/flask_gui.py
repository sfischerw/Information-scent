from flask import Flask
from flask import render_template
import json

app = Flask(__name__)

@app.route('/')

def ia_view():


    mon_titre = "GUI"

    gui3 = { "name": "Home", 
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

    
    #size = "size"

    gui4 = { "name": "home", "children":[
       { "name": "car",
       "children":[
          { "name": "air conditioning",
          "children":[
             { "name": "ventilation settings",
              "size": 20},
             { "name": "recycling mode",
              "size": 20},
             { "name": "increase celsius degrees",
              "size": 20},
    ]},
          { "name": "driving assistance",
          "children":[
             { "name": "cruise control",
             "children":[
                { "name": "activate cruise control",
                 "size": 20},
                { "name": "turn off cruise control",
                 "size": 20},
    ]},
             { "name": "anti theft notification",
              "size": 20},
             { "name": "lane change alert",
              "size": 20},
             { "name": "gps",
              "size": 20},
    ]},
    ]},
       { "name": "entertainment",
       "children":[
          { "name": "gaming",
           "size": 20},
          { "name": "television",
           "size": 20},
          { "name": "play radio",
           "size": 20},
    ]},
       { "name": "phone",
       "children":[
          { "name": "contact list",
           "size": 20},
          { "name": "dial",
           "size": 20},
          { "name": "check voice mail messages",
           "size": 20},
    ]},
    ]}
    
  
    guiX = { "name": "home", "children":[
       { "name": "food",
       "children":[
          { "name": "meat",
           "size": 20},
          { "name": "salad",
           "size": 20},
        ]},
           { "name": "drink",
           "children":[
              { "name": "coffee",
              "children":[
                 { "name": "conventional",
                  "size": 20},
                 { "name": "orga",
                  "size": 20},
        ]},
        ]},
           { "name": "misc",
           "children":[
              { "name": "call Ej",
              "children":[
                 { "name": "give present",
                  "size": 20},
                 { "name": "book massage",
                 "children":[
                    { "name": "check agenda",
                     "size": 20},
                    { "name": "find groupon",
                     "size": 20},
        ]},
        ]},
        ]},
        ]}

    gui5 =  { "name": "car",
       "children":[
          { "name": "air conditioning",
          "children":[
             { "name": "ventilation settings",
              "size": 20},
             { "name": "continuous",
              "size": 20},
             { "name": "pulse",
              "size": 20},
          ]},
          { "name": "filter settings",
          "children":[
             { "name": "recycle interior air",
              "size": 20},
             { "name": "pollen mode",
              "size": 20},
             { "name": "charcoal mode",
              "size": 20},
          ]},
          { "name": "temperature settings",
          "children":[
             { "name": "display current temperature",
              "size": 20},
             { "name": "colder",
              "size": 20},
             { "name": "hotter",
              "size": 20},
          ]},
         { "name": "driving assistance",
         "children":[
            { "name": "cruise control",
            "children":[
               { "name": "activate",
                "size": 20},
               { "name": "turn off",
                "size": 20},
           ]},
            { "name": "anti theft protection",
            "children":[
               { "name": "stop vehicle",
                "size": 20},
               { "name": "choose notification recipient",
                "size": 20},
               { "name": "phone pairing tracking",
                "size": 20},
            ]},
            { "name": "lane change alert",
            "children":[
               { "name": "vibrate pedal",
                "size": 20},
               { "name": "vibrate rear view mirrors",
                "size": 20},
               { "name": "vibrate steering wheel",
                "size": 20},
            ]},
            { "name": "gps",
            "children":[
               { "name": "recent destination",
                "size": 20},
               { "name": "enter destination",
                "size": 20},
               { "name": "check current coordinates",
                "size": 20},
            ]},
            ]},
         { "name": "phone",
         "children":[
            { "name": "pay bills",
             "size": 20},
            { "name": "contact lists",
            "children":[
               { "name": "work",
                "size": 20},
               { "name": "emergency",
                "size": 20},
               { "name": "family",
                "size": 20},
            ]},
            { "name": "dial",
             "size": 20},
            { "name": "voice mail",
            "children":[
               { "name": "change greetings",
                "size": 20},
               { "name": "listen messages",
                "size": 20},
               { "name": "erase last message",
                "size": 20},
            ]},
            ]},
         { "name": "entertainment",
         "children":[
            { "name": "gaming",
            "children":[
               { "name": "online apps",
                "size": 20},
               { "name": "poker",
                "size": 20},
               { "name": "chess",
                "size": 20},
            ]},
            { "name": "television",
            "children":[
               { "name": "tv series",
                "size": 20},
               { "name": "documentaries",
                "size": 20},
               { "name": "movies sorted",
               "children":[
                  { "name": "genre",
                   "size": 20},
                  { "name": "rating",
                   "size": 20},
                  { "name": "release date",
                   "size": 20},
            ]},
            ]},
            { "name": "radio",
            "children":[
               { "name": "electronic",
                "size": 20},
               { "name": "pop",
                "size": 20},
               { "name": "classic",
                "size": 20},
            ]},
            ]},
            ]}


    return render_template('index.html', title=mon_titre, gui=json.dumps(gui5))

if __name__ == '__main__':
    app.run(debug=False)



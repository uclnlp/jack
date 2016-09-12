from flask import Flask
from flask import render_template
from flask_socketio import SocketIO
import os
from quebap.projects.playqa.nein.agent import *
from flask import request


class App:
    """
    A Flask WebApp for interaction with a nein.Agent.
    """

    def __init__(self, agent, debug=False):
        self.app = Flask(__name__)
        self.app.debug = debug
        self.app.config['SECRET_KEY'] = 'secret!'
        self.socketio = SocketIO(self.app)
        self.actions = [':read', ':init', ':help', ':quit']
        home_dir = os.path.expanduser('~')
        nein_dir = home_dir + '/.nein'
        if not os.path.exists(nein_dir):
            os.makedirs(nein_dir)
        self.history_file = open(nein_dir + '/history.txt', 'a+')
        self.history_file.seek(0)
        self.history = [line.rstrip() for line in self.history_file.readlines()]

        @self.app.route("/")
        def shell():
            return render_template("shell.html")

        def update_client():
            # socketio.emit('vocab', {'actionTypes': ['exit'], 'pred1s': [], 'pred2s': [], 'inputs': []})
            def rev(input):
                result = list(input)
                result.reverse()
                return result

            self.socketio.emit('vocab', {'actionTypes': self.actions, 'pred1s': [], 'pred2s': [], 'inputs': []})
            self.socketio.emit('history', rev(self.history))  # TODO: better to send history *deltas*

        @self.socketio.on('user_input')
        def handle_message(json):
            input = json['code']
            if len(self.history) > 0 and self.history[-1] != input:
                self.history.append(input)
                self.history_file.write(input + '\n')
                self.history_file.flush()
            output = agent(input)
            print('received message: ' + str(output.scores))

            if output.status is RESPONSE:
                self.socketio.emit('result',
                                   {'status': output.status,
                                    'response': output.response,
                                    'scores': output.scores[0].tolist(),
                                    'candidates': output.candidates})
                self.socketio.emit('provenance', output.provenance.__dict__)

            elif output.status is ACK:
                self.socketio.emit('ack', {'response': output.response})

            elif output.status is HELP:
                self.socketio.emit('help', output.__dict__)

            elif output.status is QUIT:
                self.socketio.emit('msg', {'msg': "Bye!"})
                func = request.environ.get('werkzeug.server.shutdown')
                if func is None:
                    raise RuntimeError('Not running with the Werkzeug Server')
                import time
                time.sleep(2)
                func()
            else:
                self.socketio.emit('msg', {'msg': output.response})

            update_client()

        @self.socketio.on('vocab?')
        def handle_message():
            update_client()
            print('Vocab')

    def run(self):
        self.socketio.run(self.app)


def agent(input):
    status = QUIT if input == QUIT else OK
    return Output(input, status)

app = App(agent)
app.run()




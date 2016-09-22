from flask import Flask
from flask import render_template
from flask_socketio import SocketIO
import os

from quebap.projects.playqa.model import Model1
from quebap.projects.playqa.nein.agent import *
from flask import request


class App:
    """
    A Flask WebApp for interaction with a nein.Agent.
    """

    def __init__(self, debug=False):
        self.app = Flask(__name__)
        self.app.debug = debug
        self.app.config['SECRET_KEY'] = 'secret!'
        self.socketio = SocketIO(self.app)
        self.actions = ['%% help', '%% quit', '%% explain', '%% forget', '%% depth']
        home_dir = os.path.expanduser('~')
        nein_dir = home_dir + '/.nein'
        if not os.path.exists(nein_dir):
            os.makedirs(nein_dir)
        self.history_file = open(nein_dir + '/history.txt', 'a+')
        self.history_file.seek(0)
        self.history = [line.rstrip() for line in self.history_file.readlines()]
        self.kb = []
        self.questions = []
        self.model = Model1(20, 3)
        self.trace = None

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
            input = json['code'].strip()
            if len(self.history) > 0 and self.history[-1] != input:
                self.history.append(input)
                self.history_file.write(input + '\n')
                self.history_file.flush()
            # output = agent(input)
            # print('received message: ' + str(output.scores))

            if input.startswith('%%'):
                command = input[2:].strip()
                if command == 'help':
                    self.socketio.emit('help', {'help_msg': [
                        ('%% help', 'this help message'),
                        ('%% quit', 'exit play-qa')]})
                elif command == 'explain':
                    self.trace = self.model.query_iteratively_decoded([self.questions[-1]], self.kb)
                    print(self.trace[0].questions.transformations)
                    # print(self.trace[0][2])
                    explanations = []
                    for step in range(0, len(self.trace)):
                        explanation = {
                            'extractions': [(self.kb[i],
                                             float(self.trace[step].extractions.transformation_probs[0][i]),
                                             " ".join(self.trace[step].extractions.transformations[0][i])) for i in
                                            range(0, len(self.kb))],
                            'questions': [(self.kb[i],
                                           float(self.trace[step].questions.transformation_probs[0][i]),
                                           " ".join(self.trace[step].questions.transformations[0][i])) for i in
                                          range(0, len(self.kb))],
                            'term_prob': float(self.trace[step].terminate_prob[0]),
                            'extraction': self.trace[step].extractions.summarized_decoded,
                            'question': self.trace[step].questions.summarized_decoded
                        }
                        explanations.append(explanation)
                    print(explanations[0])
                    self.socketio.emit('explain', explanations)
                elif command == "quit":
                    self.socketio.emit('msg', {'msg': "Bye!"})
                    func = request.environ.get('werkzeug.server.shutdown')
                    if func is None:
                        raise RuntimeError('Not running with the Werkzeug Server')
                    import time
                    time.sleep(2)
                    func()
                elif command == "forget":
                    self.kb = []
                    self.socketio.emit('msg', {'msg': "Already forgotten!"})
                elif command.startswith("depth"):
                    depth = int(command.split()[1])
                    self.model = Model1(20, depth)
                    self.socketio.emit('msg', {'msg': "Depth: {}".format(depth)})



            # if output.status is RESPONSE:
            #     self.socketio.emit('result',
            #                        {'status': output.status,
            #                         'response': output.response,
            #                         'scores': output.scores[0].tolist(),
            #                         'candidates': output.candidates})
            #     self.socketio.emit('provenance', output.provenance.__dict__)
            #
            # elif output.status is ACK:
            #     self.socketio.emit('ack', {'response': output.response})
            #
            # elif output.status is HELP:
            #     self.socketio.emit('help', output.__dict__)

            else:
                if "?" in input:
                    question, answer = input.split("?")
                    self.questions.append(question)
                    self.trace = self.model.query_iteratively_decoded([question], self.kb)
                    self.socketio.emit('msg', {'msg': self.trace[-1].extractions.summarized_decoded})
                else:
                    self.kb.append(input)
                    self.socketio.emit('msg', {'msg': 'Understood'})

            update_client()

        @self.socketio.on('vocab?')
        def handle_message():
            update_client()
            print('Vocab')

    def run(self):
        self.socketio.run(self.app)


app = App()
app.run()

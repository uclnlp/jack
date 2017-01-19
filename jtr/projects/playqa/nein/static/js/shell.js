$(function () {
        "use strict";

        //setupWebLog();

        //shell eval web socket
        var transport = 'websocket';
        var socket = io.connect('http://' + document.domain + ':' + location.port);
        //var socket = $.atmosphere;
        //var request = {
        //    url: "/shell-eval",
        //    contentType: "application/json",
        //    logLevel: 'debug',
        //    transport: transport,
        //    fallbackTransport: 'long-polling'
        //};


        var editor = ace.edit("aceEditor");
        editor.setTheme("ace/theme/chrome");
        var TextMode = require("ace/mode/text").Mode;
        //require.undef("ShellHighlightRules");

        var Rules = require("ShellHighlightRules").ShellHighlightRules;
        var shellMode = new TextMode();
        shellMode.HighlightRules = Rules(undefined);
        editor.getSession().setMode(shellMode);
        editor.setOptions({
            maxLines: Infinity,
            showGutter: false,
            cursorStyle: "wide",
            highlightActiveLine: false,
//        useWrapMode:true,
            enableBasicAutocompletion: true
            //enableLiveAutocompletion: true

        });
        editor.getSession().setUseWrapMode(true);

        var acelangTools = ace.require("ace/ext/language_tools");

        function toCompletions(names, meta) {
            return names.map(function (name) {
                return {name: "Name", value: name, score: 100, meta: meta}
            });
        }

        function removeSharedPrefix(completion, prefix, line, column) {
            //find the first occurrence of the prefix in the completion
            var prefixIndex = 0;
            for (; (prefixIndex < completion.length - prefix.length) &&
                   completion.substring(prefixIndex, prefixIndex + prefix.length) !== prefix; prefixIndex++) {
            }
            var linePrefix = line.substring(column - prefix.length - prefixIndex, column - prefix.length);
            return linePrefix === completion.substring(0, prefixIndex) ?
                completion.substring(prefixIndex) :
                completion
            //line: ar
            //prefix: ar
            //full completion: learn text is text
            //partial completion: arn text is text
            //

            //we only cut off the completion if the line matches
            return result;
            //then go backwards while the completion matches the line, and remove everything until it doesn't match.
        }

        function removeSharedPrefixes(completions, prefix, line, column) {
            return completions.map(function (c) {
                return removeSharedPrefix(c, prefix, line, column);
            });
        }

        var shellCompleter = {
            getCompletions: function (editor, session, pos, prefix, callback) {
                var line = editor.session.getLine(pos.row).slice(0, pos.column);
                var token = editor.session.getTokenAt(pos.row, pos.column);
                console.log(token);
                var keywords = toCompletions(["if", "learn", "then", "explain"], "keyword");
                console.log(prefix);
                console.log(line);
                console.log(pos);
                //console.log(removeSharedPrefix("Add to sheet","to","Add to",6));
                if (vocab !== undefined) {
                    var pred1s = toCompletions(removeSharedPrefixes(vocab.pred1s, prefix, line, pos.column), "property");
                    var pred2s = toCompletions(removeSharedPrefixes(vocab.pred2s, prefix, line, pos.column), "verb");
                    var actions = toCompletions(removeSharedPrefixes(vocab.actionTypes, prefix, line, pos.column), "actionTypes");
                    var inputs = toCompletions(removeSharedPrefixes(vocab.inputs, prefix, line, pos.column), "inputs");
                    var histories = toCompletions(removeSharedPrefixes(history, prefix, line, pos.column), "history");
                    var all = keywords.concat(pred1s, pred2s, actions, inputs, histories);
                    callback(null, all);
                } else {
                    callback(null, keywords);
                }
            }
        };
        acelangTools.addCompleter(shellCompleter);

        var vocab;
        var history = [];
        var historyIndex = 0;

        function createResultHTML(result) {
            console.log("creating result");
            console.log(result);
            if (result.length === 0) return "false";
            if (result.length === 1 && Object.keys(result[0].map).length === 0) return "true";
            var table = $('<table class="bindings"/>');

            //create header
            var thead = $("<thead/>").appendTo(table);
            var headRow = $("<tr/>").appendTo(thead);
            var headBinding = result[0].map;
            for (var head_variable in headBinding) {
                if (headBinding.hasOwnProperty(head_variable)) {
                    headRow.append('<th>' + head_variable + '</th>');
                }
            }
            var tbody = $("<tbody/>").appendTo(table);
            for (var i = 0; i < result.length; i++) {
                var row = $("<tr/>").appendTo(tbody);
                var binding = result[i].map;
                for (var variable in binding) {
                    if (binding.hasOwnProperty(variable)) {
                        row.append('<td>' + binding[variable] + '</td>');
                    }
                }
                console.log(row);
            }
            console.log(table);
            return table[0].outerHTML;

        }

        socket.on('connect', function () {
            console.log("Opened connection");
            socket.emit('vocab?');
        });

        //TODO: transform to socket.io
        socket.on('vocab', function (server_vocab) {
            console.log("Received Vocab");
            console.log(vocab);
            var shellMode = new TextMode();
            vocab = server_vocab;
            shellMode.HighlightRules = Rules(vocab);
            editor.getSession().setMode(shellMode);
        });

        function show_response_and_move_on(resultHTML) {
            var code = editor.getValue();
            var li = '<li><div class="oldPrompt"> ></div><div class="command_and_result">' +
                '<div class="oldCommand">' + code + '</div><div class="result">' + resultHTML + '</div></div></li>';
            $("#editorItem").before(li);
            editor.setValue("");
            $("#inner-left").scrollTop($("#inner-left").prop("scrollHeight"));
        }

        console.log("Test12");
        socket.on('explain', function (result) {

            function create_table(sorted_rows, header) {
                var table = $('<table class="bindings"/>');
                var tbody = $("<tbody/>").appendTo(table);
                for (var i = 0; i < sorted_rows.length; i++) {
                    var row = $("<tr/>").appendTo(tbody);
                    row.append('<td>' + sorted_rows[i][0] + '</td>');
                    row.append('<td>' + sorted_rows[i][1].toFixed(3) + '</td>');
                    row.append('<td>' + sorted_rows[i][2] + '</td>');
                    console.log(row);
                }
                return "<h4>" + header + "</h4>" + table[0].outerHTML;
            }

            var full_response = "";

            for (var step = 0; step < result.length; step++) {

                var sorted_rows = result[step].extractions.sort(function (a, b) {
                    return b[1] - a[1]
                });
                var sorted_questions = result[step].questions.sort(function (a, b) {
                    return b[1] - a[1]
                });

                var response =
                    "<h3> Step " + step + "</h3>" +
                    "Answer: " + result[step].extraction + "<br>" +
                    "Question: " + result[step].question + "<br>" +
                    "Termination Prob: " + result[step].term_prob.toFixed(3);
                full_response += response + create_table(sorted_rows, "Answers") + create_table(sorted_questions, "Questions")
            }
            console.log("Yo!");
            show_response_and_move_on(full_response);
        });

        socket.on('result', function (result) {
            var code = editor.getValue();
            var exponentiated = result.scores.map(Math.exp);
            var norm = exponentiated.reduce(function (pv, cv) {
                return pv + cv;
            }, 0.0);
            var zipped = exponentiated.map(function (e, i) {
                return [result.candidates[i], exponentiated[i] / norm];
            });
            var sorted = zipped.sort(function (a, b) {
                return b[1] - a[1]
            });
            var table = $('<table class="bindings"/>');
            var tbody = $("<tbody/>").appendTo(table);
            for (var i = 0; i < sorted.length; i++) {

                var row = $("<tr/>").appendTo(tbody);
                row.append('<td>' + sorted[i][0] + '</td>');
                row.append('<td>' + sorted[i][1].toFixed(3) + '</td>');
                console.log(row);
            }
            console.log(table);
            var tableHTML = table[0].outerHTML;
            show_response_and_move_on(tableHTML);
        });


        socket.on('ack', function (result) {
            show_response_and_move_on(result.response);
        });

        socket.on('help', function (result) {
            //print(result)
            var rawHTML = "<table>\n";
            for (var i = 0; i < result.help_msg.length; ++i) {
                var command = result.help_msg[i][0];
                var explain = result.help_msg[i][1];
                rawHTML += '<tr><td class="ace_action">' + command + '</td><td>' + explain + '</td></tr>\n'
            }
            rawHTML += "</table>";
            show_response_and_move_on(rawHTML);
        });

        socket.on('msg', function (result) {
            show_response_and_move_on(result.msg);
        });


        socket.on('history', function (server_history) {
            console.log("Received History");
            console.log(history);
            history = server_history;
        });

        socket.on('provenance', function (provenance) {
            console.log("Received provenance");
            console.log(provenance);
            var template = $('#template_sentence');
            var list = $('#sentences');
            list.empty();
            for (var i = 0; i < provenance.sentences.length; i++) {
                var newSentence = template.clone();
                $('.text', $(newSentence)).text(provenance.sentences[i]);
                //$('.text', $(newSentence)).text("The > combinator separates two selectors and matches only those elements matched by the second selector that are direct children of elements matched by the first. By contrast, when two selectors are combined");
                $('.attention', $(newSentence)).text(provenance.attention[i].toFixed(2));
                //newSentence.text(provenance.sentences[i]);
                newSentence.show();
                newSentence.attr("id", "sent" + i);
                list.append(newSentence);
            }
            //$('#provenance').text(provenance.sentences);
        });


        //request.onMessage = function (rs) {
        //    console.log("onMessage");
        //    console.log(rs);
        //    var code = editor.getValue();
        //    var json = jQuery.parseJSON(rs.responseBody);
        //    if (json.type === "result") {
        //        var msg = json.result;
        //        var resultHTML;
        //        if (json.resultType === "bindings") {
        //            resultHTML = createResultHTML(jQuery.parseJSON(msg));
        //        } else {
        //            resultHTML = msg;
        //        }
        //        var li = '<li><div class="oldPrompt"> ></div><div class="command_and_result">' +
        //            '<div class="oldCommand">' + code + '</div><div class="result">' + resultHTML + '</div></div></li>';
        //        $("#editorItem").before(li);
        //        editor.setValue("");
        //        $("#inner-left").scrollTop($("#inner-left").prop("scrollHeight"));
        //    } else if (json.type === "web-log") {
        //        $("#web-log").append(json.msg + '\n');
        //    } else if (json.type === "vocab") {
        //        vocab = json;
        //        console.log("received vocab");
        //        console.log(vocab);
        //        var shellMode = new TextMode();
        //        shellMode.HighlightRules = Rules(vocab);
        //        editor.getSession().setMode(shellMode);
        //    } else if (json.type === "error") {
        //        var msg = json.result;
        //        $("#editorItem").before(
        //            '<li><div class="oldPrompt"> ></div><div class="command_and_result">' +
        //            '<div class="oldCommand">' + code + '</div><div class="error">' + msg + '</div></div></li>');
        //        editor.setValue("");
        //    } else if (json.type == "history") {
        //        history = json.history;
        //    }
        //
        //};

        //var subSocket = socket.subscribe(request);

        //TODO: adapt to socket.io
        editor.commands.addCommand({
            name: "runCode",
            bindKey: {win: "Enter", mac: "Enter"},
            exec: function (editor) {
                var code = editor.getValue();
                if (code != "") {
                    var json = {
                        code: code,
                        type: "code"
                    };
                    socket.emit('user_input', json);
                    //subSocket.push(jQuery.stringifyJSON(json));
                    historyIndex = 0;
                    //once the result comes back, do this operation
                }
                console.log("Enter pressed")
            }
        });

        editor.commands.addCommand({
            name: "historyUp",
            bindKey: {win: "Up", mac: "Up"},
            exec: function (editor) {
                if (historyIndex < history.length) {
                    editor.setValue(history[historyIndex]);
                    historyIndex += 1;
                }
                console.log("Up!")
                console.log(history)
            }
        });

        editor.commands.addCommand({
            name: "historyDown",
            bindKey: {win: "Down", mac: "Down"},
            exec: function (editor) {
                if (historyIndex > 0) {
                    historyIndex -= 1;
                    editor.setValue(history[historyIndex]);
                } else if (historyIndex == 0) {
                    editor.setValue("");
                }
                console.log(historyIndex);
                console.log("Down!");
                console.log(history)

            }
        });

        $("#this").keypress(function (e) {
            if (e.which == 13) {
                console.log("Enter!");
                subSocket.push(jQuery.stringifyJSON({type: "webTextInput", current: $("#this").val()}));
            }
        });

        function notifyWebTextInputServer() {
            subSocket.push(jQuery.stringifyJSON({type: "webTextInput", current: $("#this").val()}));
        }

        $("#this").bind("enterKey", function (e) {
            console.log("Enter!");
            notifyWebTextInputServer();
        });
        $("#this-button").click(function (e) {
            console.log("Click!");
            notifyWebTextInputServer();
        });


    }
);


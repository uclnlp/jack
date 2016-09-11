define("ShellHighlightRules", [], function (require, exports, module) {
    "use strict";

    var oop = require("ace/lib/oop");
    var TextHighlightRules = require("ace/mode/text_highlight_rules").TextHighlightRules;

    console.log("Why?")

    var never = "(?!x)x"

    function funId(next) {
        return {
            token: "function",
            regex: "it's true that|the value of|log",
            next: next
        }
    }

    function funApp(grammar, name, next, stop) {
        grammar[name] = [
            funId(name + 'withFun'),

            {
                token: "freetext",
                regex: "[^\\s\$]+",
            },
            {
                token: "keyword",
                regex: stop,
                next: next
            },
            {
                token: "slot",
                regex: "\\$\\w+"
            }
        ];
        grammar[name + 'withFun'] = [
            {
                token: "freetext",
                regex: "[^\\s\$]+",
            },
            {
                token: "keyword",
                regex: stop,
                next: next
            },
            {
                token: "slot",
                regex: "\\$\\w+"
            }
        ]
    }

    function arrayToRegex(array) {
        return array.reduce(function (previousValue, currentValue, index, array) {
            return previousValue + '|' + currentValue
        }, never);
    }

    var ShellHighlightRules = function (vocab) {

        var ruleFunction = function () {
            // regexp must not have capturing parentheses. Use (?:) instead.
            // regexps are ordered -> the first match is used

            var pred1s = vocab === undefined ? never : arrayToRegex(vocab.pred1s);
            var pred2s = vocab === undefined ? never : arrayToRegex(vocab.pred2s);
            var actions = vocab === undefined ? never : arrayToRegex(vocab.actionTypes);
            var inputs = vocab === undefined ? never : arrayToRegex(vocab.inputs);

            console.log(pred1s);

            var freetext = "\".*?\"|[^\\s\$\?\"]+";

            function action(grammar, name, next, stop) {
                grammar[name] = [
                    {
                        token: "keyword",
                        regex: stop,
                        next: next
                    },
                    {
                        token: "slot",
                        regex: "\\$\\w+"
                    },
                    {
                        token: "freetext",
                        regex: freetext
                    }];

            }


            function atom(grammar, name, next, stop) {
                grammar[name] = [
                    {
                        token: "keyword",
                        regex: stop,
                        next: next
                    },
                    {
                        token: "pred2",
                        regex: pred2s,
                        next: name + '2'
                    },
                    {
                        token: "pred1",
                        regex: pred1s,
                        next: next
                    },
                    {
                        token: "slot",
                        regex: "\\$\\w+"
                    },
                    {
                        token: "freetext",
                        regex: freetext
                    }];

                grammar[name + '2'] = [
                    {
                        token: "keyword",
                        regex: stop,
                        next: next
                    },
                    {
                        token: "slot",
                        regex: "\\$\\w+"
                    },
                    {
                        token: "freetext",
                        regex: freetext
                    }
                ];

            }

            this.$rules = {
                "start": [
                    {
                        token: "keyword",
                        regex: "if",
                        next: "ifStart"
                    },
                    {
                        token: "keyword",
                        regex: "explain",
                        next: "atomStart"
                    },
                    {
                        token: "action",
                        regex: actions,
                        next: "actionStart"
                    },
                    {
                        token: "freetext",
                        regex: freetext,
                        next: "atomStart"
                    },
                    {
                        token: "slot",
                        regex: "\\$\\w+",
                        next: "atomStart",
                    }
                ],
                "action": [
                    {
                        token: "action",
                        regex: actions,
                        next: "actionStart"
                    },
                ],
                "illegal": [
                    {
                        token: "illegal",
                        regex: ".*"
                    }
                ]
            };
            atom(this.$rules, "atomStart", "illegal", "\\?");
            action(this.$rules, "actionStart", "illegal", never);
            atom(this.$rules, "ifStart", "action", "then");


        };
        oop.inherits(ruleFunction, TextHighlightRules);
        return ruleFunction;
    };


    exports.ShellHighlightRules = ShellHighlightRules;

});
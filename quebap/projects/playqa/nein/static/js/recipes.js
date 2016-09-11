/**
 * Created by riedel on 21/09/15.
 */
$(function () {
    var template = $("#template");
    //template.hide();

    var vocab;
    var transport = 'websocket';
    var socket = $.atmosphere;
    var request = {
        url: "/shell-eval",
        contentType: "application/json",
        logLevel: 'debug',
        transport: transport,
        fallbackTransport: 'long-polling'
    };

    request.onOpen = function (response) {

        console.log("Opened " + response);
        subSocket.push("Vocab?");
        subSocket.push("Rules?");
    };

    request.onMessage = function (rs) {
        console.log("onMessage");
        console.log(rs);
        var json = jQuery.parseJSON(rs.responseBody);
        console.log(json);

        if (json.type === "rules") {
            var rules = jQuery.parseJSON(json.result).rules;
            for (var i = 0; i < rules.length; ++i) {
                var rule = rules[i];
                var clone = template.clone();
                clone.attr("id", "recipe" + i);
                template.parent().append(clone);
                $(".if .verb", clone).val(rule.query.pred);
                $(".if .subject", clone).val(rule.query.arg1.text);
                $(".then .object", clone).val(rule.action.arg.text);
                if (vocab !== undefined && $.inArray(rule.query.pred, vocab.pred1s) != -1)
                    $(".if .object", clone).hide();
                //setup listeners
            }
        } else if (json.type === "vocab") {
            vocab = json;
            console.log("received vocab");
            console.log(vocab);
        }


    };

    var subSocket = socket.subscribe(request);


});



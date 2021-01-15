$(document).ready(function() {
    console.log("document ready");
    
    results_dir = "../results/";
    benchmarks = [];
    get_benchmarks();
    
    case_colors = {
        "TRUE_POSITIVE": "green",
        "FALSE_POSITIVE": "red",
        "FALSE_NEGATIVE": "orange",
        "UNDETECTED": "orange",
        "WAS_DETECTED": "aqua"
    };
    
    $("#select_benchmark").change(function() {
        get_approaches();
    });
    $("#select_subset").change(function() {
        get_approaches();
    });
    
    $("#select_approach").change(function() {
        show_results();
    });
    
    current_tab = "details";
    $("#tab_button").click(function() {
        $("#tab_button").html(current_tab);
        if (current_tab == "details") {
            current_tab = "sequences";
            $("#sequences_tab").show();
            $("#details_tab").hide();
            show_sequence_comparison();
        } else {
            current_tab = "details";
            $("#details_tab").show();
            $("#sequences_tab").hide();
        }
    });
    
    $("#previous_sequence").click(function() {
        value = $("#sequence_number").val()
        value = parseInt(value);
        $("#sequence_number").val(value - 1);
        show_sequence_comparison();
    });
    
    $("#next_sequence").click(function() {
        value = $("#sequence_number").val()
        value = parseInt(value);
        $("#sequence_number").val(value + 1);
        show_sequence_comparison();
    });
    
    $("#sequence_number").on("change", function() {
        sequence_number = $("#sequence_number").val();
        show_sequence_comparison();
    });
});

function get_benchmarks() {
    $.get(results_dir, function(data) {
        $(data).find("a").each(function() {
            name = $(this).attr("href");
            name = name.substring(0, name.length - 1);
            benchmarks.push(name);
	        console.log(name);
            $("#select_benchmark").append(new Option(name, name));
        });
        $("#select_benchmark").prop("selectedIndex", -1);
    });
}

function get_approaches() {
    benchmark = $("#select_benchmark option:selected").val();
    console.log(benchmark);
    
    subset = $("#select_subset option:selected").val();
    console.log(subset);
    
    results_folder = results_dir + benchmark + "/" + subset + "/";
    approaches = [];
    $("#select_approach").empty();
    $.get(results_folder, function(data) {
        $(data).find("a").each(function() {
            file_name = $(this).attr("href");
            if (file_name.endsWith(".results.json")) {
                approach_name = file_name.substring(0, file_name.length - 13);
                approaches.push(approach_name);
                $("#select_approach").append(new Option(approach_name, approach_name));
            }
        });
        $("#select_approach").prop("selectedIndex", -1);
        $("#details_table").html("select an approach above");
        $("#sequences_table").html("");
        show_benchmark_results();
    });
}

function show_benchmark_results() {
    get_benchmark_results();
}

function percent(fraction) {
    return (fraction * 100).toFixed(2);
}

function get_benchmark_results() {
    benchmark_results = {};
    approaches.forEach(function(approach, i) {
        promise = new Promise(function(add_results) {
            results_file = results_folder + approach + ".results.json";
            $.get(results_file, function(data) {
                add_results([approach, data]);
            });
        });
        promise.then(function([approach, results]) {
            benchmark_results[approach] = results;
            if (Object.keys(benchmark_results).length == approaches.length) {
                create_benchmark_results_table();
            }
        });
    });
    if (approaches.length == 0) {
        create_benchmark_results_table();
    }
}

function results_table_header(details) {
    if (details) {
        col1_title = "error type";
    } else {
        col1_title = "approach";
    }
    header = "<tr>";
    header += "<th rowspan=\"2\">" + col1_title + "</th>";
    header += "<th colspan=\"3\">detection</th>";
    header += "<th colspan=\"3\">correction</th>";
    if (!details) {
        header += "<th rowspan=\"2\">sequence accuracy</th>";
    }
    header += "</tr>";
    
    header += "<tr>";
    for (i = 0; i < 2; i++) {
        header += "<th>precision</th>";
        header += "<th>recall</th>";
        header += "<th>F1</th>";
    }
    header += "</tr>\n";
    return header;
}

function table_row(entries) {
    row = "<tr>";
    for (entry of entries) {
        row += "<td>" + entry + "</td>";
    }
    row += "</tr>\n";
    return row
}

function create_benchmark_results_table() {
    table = "<table>\n";
    table += results_table_header(false);
    for (approach of approaches) {
        results = benchmark_results[approach];
        console.log(approach);
        console.log(results);
        table += table_row([approach,
                            percent(results.all.detection.precision),
                            percent(results.all.detection.recall),
                            percent(results.all.detection.f1),
                            percent(results.all.correction.precision),
                            percent(results.all.correction.recall),
                            percent(results.all.correction.f1),
                            percent(results.accuracy)]);
    }
    table += "</table>";
    $("#benchmark_table").html(table);
}

function sequence_table_header(col1_title) {
    header = "<tr>";
    header += "<th>" + col1_title + "</th>";
    header += "<th>MISSPELLED</th>";
    header += "<th>PREDICTED</th>";
    header += "<th>GROUND TRUTH</th>";
    header += "</tr>\n";
    return header;
}

function display_token_cases(tokens) {
    display_html = "";
    for (i in tokens) {
        token = tokens[i];
        if (i > 0) {
            display_html += " ";
        }
        if (token["case"] == null) {
            display_html += token["text"];
        } else {
            color = case_colors[token["case"]];
            labels = token["error_type"] + ", " + token["predicted_label"] + ", " + token["case"];
            open_tag = "<span style=\"background-color:" + color + "\" title=\"" + labels + "\">";
            close_tag = "</span>";
            display_html += open_tag + token["text"] + close_tag;
        }
    }
    return display_html;
}

function sequence_table_row(col1_entry, evaluated_sequence) {
    row = table_row([col1_entry,
                     display_token_cases(evaluated_sequence.misspelled_tokens),
                     display_token_cases(evaluated_sequence.predicted_tokens),
                     display_token_cases(evaluated_sequence.correct_tokens)]);
    return row;
}

function show_results() {
    approach = $("#select_approach option:selected").val();
    show_sequences();
    create_detail_table();
}

function create_detail_table(path) {
    results = benchmark_results[approach];
    details_table = "<table>\n";
    details_table += results_table_header(true);
    for (label of ["all", "NONWORD", "REAL_WORD", "SINGLE_EDIT", "MULTI_EDIT", "SPLIT", "MERGE", "MIXED"]) {
        details_table += table_row([label,
                                    percent(results[label].detection.precision),
                                    percent(results[label].detection.recall),
                                    percent(results[label].detection.f1),
                                    percent(results[label].correction.precision),
                                    percent(results[label].correction.recall),
                                    percent(results[label].correction.f1)]);
    }
    details_table += "</table>\n";
    details_table += "<br>sequence accuracy = " + percent(results.accuracy) + " %";
    $("#details_table").html(details_table);
}

function show_sequences() {
    sequences_path = results_folder + approach + ".sequences/";
    table = "<table>\n" + sequence_table_header("#");
    j = 0;
    n = benchmark_results[approach].sequences;
    for (i = 0; i < n; i++) {
        promise = new Promise(function(add_line) {
            sequence_file = sequences_path + i + ".json";
            console.log(sequence_file);
            $.get(sequence_file, function(data) {
                add_line(data);
            });
        });
        promise.then(function(data) {
            table += sequence_table_row(j, data);
            j += 1;
            if (j == n) {
                table += "</table>"
                $("#sequences_table").html(table);
            }
        });
    }
}

function show_sequence_comparison() {
    sequence_number = $("#sequence_number").val();
    let promises = [];
    let evaluated_sequences = {};
    approaches.forEach(function(approach, i) {
        promise = new Promise(function(add_sequence) {
            results_file = results_folder + approach + ".sequences/" + sequence_number + ".json";
            $.get(results_file, function(data) {
                add_sequence([approach, data]);
            });
        });
        promise.then(function([approach, evaluated_sequence]) {
            evaluated_sequences[approach] = evaluated_sequence;
        });
        promises.push(promise);
    });
    Promise.all(promises).then(function() {
        table = "<table>\n";
        table += sequence_table_header("approach");
        for (approach of approaches) {
            table += sequence_table_row(approach, evaluated_sequences[approach]);
        }
        table += "</table>";
        $("#sequences_comparison").html(table);
    });
}

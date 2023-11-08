$(function () {
    $('select#event').on('change', function () {
        $.ajax({
            url: "/sensoranalysis/qaselect",
            type: "GET",
            contentType: 'application/json;charset=UTF-8',
            data: {
                'event': document.getElementById('event').value
            },
            dataType: "json",
            success: function (data) {
                $('#topicgraph').empty();
                $('#topicdist').empty();
                $('#topictable').empty();
                $('#trendtable').empty();
                $('#linegraph').empty();
            }
        });
    });
});

$(function () {
	//$(document).on('click', '#btnLbl', function () {
	$("#dwnButton").unbind().click(function() {
				topbar.show()
				
				$.ajax({
                            url: '/sensoranalysis/getKeywords2',
                            type: 'GET',
                            contentType: 'application/json;charset=UTF-8',
                            data: {
                                'selected_event': document.getElementById('phone').value
                            },
                            dataType: "json",
                            success: function (data) {
								console.log("file downloaded complete");
								console.log(data);
								console.log(data.length)
								var x= document.getElementById('randomEventSelection');
								let keys = Object.keys(data);
								let vals = Object.values(data);
								
								console.log(keys);
								console.log(vals[0]);
								
								for (var i = 0; i<keys.length; i++){
									var opt = document.createElement('option');
									opt.value = keys[i];
									opt.innerHTML = vals[i];
									
									x.add(opt);
								}
								topbar.hide()
								showAlert("The events related to the keywords have been downloaded!");
                                //$('#hcsg').empty();
                                //Plotly.newPlot('hcsg', data)
                            }
                        });
			});
});

//$(function () {
function handleBarrier(button){
    //$('select#barrier').on('change', function () {
        $.ajax({
            url: "/sensoranalysis/qaselect",
            type: "GET",
            contentType: 'application/json;charset=UTF-8',
            data: {
                'barrier': document.getElementById('barrier').value
            },
            dataType: "json",
            success: function (data) {
                $('#topicgraph').empty();
                $('#topicdist').empty();
                $('#topictable').empty();
                $('#trendtable').empty();
                $('#linegraph').empty();
            }
        });
    };//);
//});


$(function () {
    $('select#barrierRadial').on('change', function () {
        $.ajax({
            url: "/sensoranalysis/generateRadialTree",
            type: "GET",
            contentType: 'application/json;charset=UTF-8',
            data: {
                'barrier': document.getElementById('barrierRadial').value
            },
            dataType: "json",
            success: function (data) {
                $('#topicgraph').empty();
                $('#topicdist').empty();
                $('#topictable').empty();
                $('#trendtable').empty();
                $('#linegraph').empty();
            }
        });
    });
});

function getSelectValues(select) {
  var result = [];
  var options = select.selectedOptions;
  var opt;
  console.log(options.length)
  for (var i=0, iLen=options.length; i<iLen; i++) {
	  opt = options[i];
	  var a = opt.value
	  console.log(a)
      result.push(a);
  }
  console.log(result.join(","));
  return result.join(",");
}

function handleClicked(button){
   // $('#QAform_1').on('click', 'button', function () {
		topbar.show()
        $.ajax({
            url: '/sensoranalysis/qaline',
            type: 'GET',
            contentType: 'application/json;charset=UTF-8',
            data: {
                  'commaValues': getSelectValues(document.getElementById('commaValues')),
                  'selBarrier': document.getElementById('barrier').value
            },
            dataType: "json",
            success: function (data) {
                var layout = {'xaxis.range': [1, 30]}
                //Plotly.newPlot('qalinegraph', data);
                //Plotly.deleteTraces('qalinegraph', 0);
                Plotly.newPlot('qalinegraph', data);
				
				$.ajax({
					url: '/sensoranalysis/qa_word_clouds2',
					type: 'GET',
					contentType: 'application/json;charset=UTF-8',
					data: {
						//'selected_event': document.getElementById('event').value,
						'commaValues': getSelectValues(document.getElementById('commaValues')),
						'selBarrier': document.getElementById('barrier').value
					},
					dataType: "json",
					success: function (data) {
						console.log("console prints")
						//console.log(data)
						try{
							console.log(data.length)
						}
						catch(e)
						{
							console.log("failed")
						}
						//console.log(data['data']['x']);
						//console.log(typeof(data['data']['x']));
						//console.log(data[0]);
						//console.log(JSON.stringify(data, null, 4));
						//console.log(data['data'].length)
						//var tot = data['data'].length
						//console.log(data['data'].length)
						//if(data == ""){
						//	document.getElementById("qawordcloud").innerHTML = "";
						//	//$('#qawordcloud').empty();
							//Plotly.deleteTraces('qawordcloud', 0);
						//	topbar.hide()
						//}else{
							
						var tot = data.length
						try{
							Plotly.deleteTraces('qawordcloud', 0);
						}
						catch(e)
						{
							console.log("failed")
						}
						document.getElementById('qawordcloud').style.height = tot*500 + 'px';
						for(x=0; x<tot;x++) {
							var board = document.createElement('div');
							var idstr = "div_"+x.toString();
							board.id = idstr;
							board.style.height = '500px';
							document.getElementById('qawordcloud').appendChild(board);
							Plotly.newPlot(idstr, data[x])
						}
						
						//
						//Plotly.newPlot('qawordcloud', data)
						topbar.hide()
					}//}
				});
            }
        });
    };//);
//}
//);

   $('#btn1').on('click', 'button', function () {
		topbar.show()
        $.ajax({
            url: '/sensoranalysis/qaline',
            type: 'GET',
            contentType: 'application/json;charset=UTF-8',
            data: {
                  'commaValues': getSelectValues(document.getElementById('commaValues')),
                  'selBarrier': document.getElementById('barrier').value
            },
            dataType: "json",
            success: function (data) {
                var layout = {'xaxis.range': [1, 30]}
                //Plotly.newPlot('qalinegraph', data);
                //Plotly.deleteTraces('qalinegraph', 0);
                Plotly.newPlot('qalinegraph', data);
				
				$.ajax({
					url: '/sensoranalysis/qa_word_clouds2',
					type: 'GET',
					contentType: 'application/json;charset=UTF-8',
					data: {
						//'selected_event': document.getElementById('event').value,
						'commaValues': getSelectValues(document.getElementById('commaValues')),
						'selBarrier': document.getElementById('barrier').value
					},
					dataType: "json",
					success: function (data) {
						console.log("console prints")
						//console.log(data)
						try{
							console.log(data['data'].length)
						}
						catch(e)
						{
							console.log("failed")
						}
						//console.log(data['data']['x']);
						//console.log(typeof(data['data']['x']));
						//console.log(data[0]);
						//console.log(JSON.stringify(data, null, 4));
						//console.log(data['data'].length)
						//var tot = data['data'].length
						//console.log(data['data'].length)
						//if(data == ""){
						//	document.getElementById("qawordcloud").innerHTML = "";
						//	//$('#qawordcloud').empty();
							//Plotly.deleteTraces('qawordcloud', 0);
						//	topbar.hide()
						//}else{
						var tot = data['data'].length							
						document.getElementById('qawordcloud').style.height = tot*500 + 'px';
						Plotly.newPlot('qawordcloud', data)
						topbar.hide()
					}//}
				});
            }
        });
    });
//});

/*$(function () {
    /*$('div#qalinegraph').on('plotly_click', function (_, data) {
        $.ajax({
            url: '/sensoranalysis/qa_analysis_table',
            type: 'GET',
            contentType: 'application/json;charset=UTF-8',
            data: {
                'date' : data.points[0].x,
                'count' : data.points[0].y,
                'legendgroup' : data.points[0].data.name,
                'selected_event': document.getElementById('event').value,
				'selBarrier': document.getElementById('barrier').value
            },
            dataType: "json",
            success: function (data) {
				Plotly.newPlot('qatrendtable', data)
            }
        });
    });
});*/

/*$(function () {
    $('div#HCform_1').on('click', 'button', function () {
		topbar.show()
        $.ajax({
            url: '/sensoranalysis/hchierarchical_clustering',
            type: 'GET',
            contentType: 'application/json;charset=UTF-8',
            data: {
				'width':document.getElementById('hclinegraph').offsetWidth
                //'selected_event': document.getElementById('event').value
            },
            dataType: "json",
            success: function (data) {
				console.log("clicked button plot")
				//if(data == "0"){
				//				document.getElementById("label").style.display = 'none';
				//				document.getElementById("slider").style.display = 'none';
				//				alert("Please select an event!")
				//				return
				//			}
				//			else{
							document.getElementById("label").style.display = 'block';
							document.getElementById("slider").style.display = 'block';
			   el = document.getElementById('hclinegraph');
	
               var layout = {
                   'xaxis.range': [2010, 2015]
                }
                //Plotly.newPlot('hclinegraph', data, layout);
                //$('#hclinegraph').empty();
				//$('#hclinegraph').append(data);
				//document.getElementById("label").style.display = 'none';
				//var elements = document.getElementsByClassName("main-svg");
				//elements[i].style.width=(2000+"px");
                Plotly.newPlot('hclinegraph', data)
				
				//Plotly.relayout('hclinegraph', {width: 1000});
				//Plotly.Plots.resize('hclinegraph')
				
				$.ajax({
				url: '/sensoranalysis/hcThemeRiver',
				type: 'GET',
				contentType: 'application/json;charset=UTF-8',
				data: {
					//'selected_event': document.getElementById('event').value
				},
				dataType: "json",
				success: function (data) {
				//$('#hcsg').empty();
                //Plotly.deleteTraces('hcsg', 0);
				//document.getElementById("label").style.display = 'none';
				//document.getElementById("slider").style.display = 'none';
                Plotly.newPlot('hcsg', data)
				topbar.hide()
				}//}
        });
            }
        });
    });
});*/


$(function () {
    $('div#HCclustering').on('plotly_click', function (_, data) {
		topbar.show()
        $.ajax({
            url: '/sensoranalysis/hcThemeRiver',
            type: 'GET',
            contentType: 'application/json;charset=UTF-8',
            data: {
                'x': data.points[0].x,
                //'selected_event': document.getElementById('event').value
            },
            dataType: "json",
            success: function (data) {
                //$('#hcsg').empty();
                //Plotly.deleteTraces('hcsg', 0);
				document.getElementById("label").style.display = 'none';
                Plotly.newPlot('hcsg', data)
				topbar.hide()
				
            }
        });
    });
});

$(function () {
    $('div#HCstreamGraph5').on('plotly_click', function (_, data) {
        //$.ajax({
        //    url: '/sensoranalysis/hc_analysis_table',
        //    type: 'GET',
        //    contentType: 'application/json;charset=UTF-8',
        //    data: {
        //        'date' : data.points[0].x,
        //        'count' : data.points[0].y,
        //        'legendgroup' : data.points[0].data.name,
        //        'selected_event': document.getElementById('event').value
        //    },
        //    dataType: "json",
        //    success: function (data) {
                //$('#topictable').append(data.table)
                //$('#trendtable').empty();
                //Plotly.deleteTraces('trendtable', 0);
                //Plotly.newPlot('trendtable', data)
        //    }
        //});
		topbar.show()
		$.ajax({
            url: '/sensoranalysis/qa_word_clouds5',
            type: 'GET',
            contentType: 'application/json;charset=UTF-8',
            data: {
                'date' : data.points[0].x,
                'count' : data.points[0].y,
                'legendgroup' : data.points[0].data.name,
                //'selected_event': document.getElementById('event').value,
            },
            dataType: "json",
            success: function (data) {
				Plotly.newPlot('hcwordcloud', data)
				topbar.hide()
            }
        });
    });
});
/////////////////////////////////////////////////////
$(function () {
    $('#form_1').on('click', 'button', function () {
		topbar.show()
        $.ajax({
            url: '/line',
            type: 'GET',
            contentType: 'application/json;charset=UTF-8',
            data: {
                'city_news': document.getElementById('city_news').value,
                'wd_list': document.getElementById('wd_list').value
            },
            dataType: "json",
            success: function (data) {
                var layout = {
                    'xaxis.range': [2010, 2015]
                }
                Plotly.newPlot('linegraph', data, layout);
				topbar.hide()
            }
        });
    });
});


$(function () {
    $('#form_2').on('click', 'button', function () {
		topbar.show()
        $.ajax({
            url: '/topic',
            type: 'GET',
            contentType: 'application/json;charset=UTF-8',
            data: {
                'news_pub': document.getElementById('news_pub').value,
                'num_k': document.getElementById('num_k').value
            },
            dataType: "json",
            success: function (data) {
                $('#topicgraph').empty();
                $('#topicgraph').append(data.lda_html);
                $.ajax({
                    url: '/topicdist',
                    type: 'GET',
                    contentType: 'application/json;charset=UTF-8',
                    data: {
                        'news_pub': document.getElementById('news_pub').value,
                        'num_k': document.getElementById('num_k').value
                    },
                    dataType: "json",
                    success: function (data) {
                        Plotly.newPlot('topicdist', data);
						topbar.hide()
                    }
                });
            }
        });
    });
});

$(function () {
    $('div#linegraph').on('plotly_click', function (_, data) {
        $.ajax({
            url: '/trendtable',
            type: 'GET',
            contentType: 'application/json;charset=UTF-8',
            data: {
                'city_news': document.getElementById('city_news').value,
                'wd_list': document.getElementById('wd_list').value,
                'year': data.points[0].x
            },
            dataType: "json",
            success: function (data) {
                $('#trendtable').empty();
                $('#trendtable').append(data.table)
            }
        });
    });
});

$(function () {
    $('div#topicdist').on('plotly_click', function (_, data) {
        $.ajax({
            url: '/topicdocs',
            type: 'GET',
            contentType: 'application/json;charset=UTF-8',
            data: {
                'news_pub': document.getElementById('news_pub').value,
                'num_k': document.getElementById('num_k').value,
                'year': data.points[0].x
            },
            dataType: "json",
            success: function (data) {
                $('#topictable').empty();
                $('#topictable').append(data.table)
            }
        });
    });
});


// function bind_button() {
//     $("#annotate").on("click", function(d) {
//         var active_entity_list = $(".entity-view.active")
//         if (active_entity_list.length != 1) {
//             alert("Please select an item")
//         }
//         var active_entity = active_entity_list.first()
//
//         // get the raw text and final annotations
//         var article = $("img").attr("src").split("/")[2];
//         var url = "annotate/" + article + "/" + active_entity.attr("data-src");
//         console.log("Calling: " + url);
//         $.ajax({
//             type: "GET",
//             url: url,
//             success: function(response) {
//                 set_page(response);
//                 bind_button();
//                 console.log("SUCESS!");
//             },
//             error: function(error_response) {
//                 console.error(error_response);
//             }
//         });
//     });
// }


//$('#first_cat').on('change',function(){
//    $.ajax({
//        url: "/bar",
//        type: "GET",
//        contentType: 'application/json;charset=UTF-8',
//        data: {
//            'selected': document.getElementById('first_cat').value
//
//        },
//        dataType:"json",
//        success: function (data) {
//            Plotly.newPlot('bargraph', data );
//        }
//    });
//});
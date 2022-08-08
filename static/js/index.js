var count = 0
$(document).ready(function(){
    $('#sw_trace').on('change', function() {
        switchStatus = $(this).is(':checked');
        console.log('Shwo Trace:'+switchStatus)
        if ($(this).is(':checked')) {
            // do display trace
            $('#vid_trace').show()
            count ++;
        }
        else {
            $('#vid_trace').hide()
        }
    })
    $('#sw_flow').on('change', function() {
        switchStatus = $(this).is(':checked');
        console.log('Shwo Flow:'+switchStatus)
        if ($(this).is(':checked')) {
            // do display flow
            $('#vid_flow').show()
        }
        else {
            $('#vid_flow').hide()
        }
    })
    $('#sw_heatmap').on('change', function() {
        switchStatus = $(this).is(':checked');
        console.log('Shwo Heatmap:'+switchStatus)
        if ($(this).is(':checked')) {
            // do display heatmap
            $('#vid_heatmap').show()
        }
        else {
            $('#vid_heatmap').hide()
        }
    })
    $('.box').hover(function(){
        $(this).css("z-index", 9)
    },function(){
        $(this).css("z-index", 1)
    })
})
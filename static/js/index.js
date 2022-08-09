var count = 0
$(document).ready(function(){
    $('#sw_trace').on('change', function() {
        switchStatus = $(this).is(':checked');
        console.log('Show Trace:'+switchStatus)
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
        console.log('Show Flow:'+switchStatus)
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
        console.log('Show Heatmap:'+switchStatus)
        if ($(this).is(':checked')) {
            // do display heatmap
            $('#vid_heatmap').show()
        }
        else {
            $('#vid_heatmap').hide()
        }
    })
    $('#sw_box').on('change', function() {
        switchStatus = $(this).is(':checked');
        console.log('Show box:'+switchStatus)
        if ($(this).is(':checked')) {
            // do display heatmap
            $('#vid_box').show()
        }
        else {
            $('#vid_box').hide()
        }
    })
    $('.box').hover(function(){
        $(this).css("z-index", 9)
    },function(){
        $(this).css("z-index", 1)
    })

})
function terminate(){
    return false
}
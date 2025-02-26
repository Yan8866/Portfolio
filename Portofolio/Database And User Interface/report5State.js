function SelectState() {
    
    var x = document.getElementById("city").value;

    $.ajax({
        url: "DisplayDataReport5.php",
        method: "POST",
        data:{
            id: x
        },
        success:function(data){
            $("#ans").html(data);
        }

    })

}
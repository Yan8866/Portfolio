<?php

/**
 * Get the database connection
 *
 * @return object Connection to the database server
 */
function getDB()
{
    $db_host = "localhost";
    $db_user = "team011";
    $bd_pass = "gatech";
    $db_name = "cs6400_su24_team011";
    
    $conn = mysqli_connect($db_host, $db_user, $bd_pass, $db_name,);

    if (mysqli_connect_error()) {
        echo mysqli_connect_error();
        exit;
    }

    return $conn;
}
?>
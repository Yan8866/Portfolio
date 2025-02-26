<?php

    $k = $_POST['id'];
    $k = trim($k);
    $conn = mysqli_connect("localhost", "team011", "gatech", "cs6400_su24_team011");

    $sql = "SELECT YEAR(SaleDate) AS TheYear, ST.StoreNumber, C.CityName, ROUND(SUM(Quantity * IFNULL(DiscountPrice, RetailPrice)), 2) AS Revenue
            FROM sale SA
            LEFT JOIN discount D
            ON SA.PID = D.PID AND SaleDate = DiscountedDate,
                product P,
                store ST,
                city C
            WHERE ST.StateName = '{$k}'
                AND C.StateName = '{$k}'
                AND ST.CityName = C.CityName
                AND SA.StoreNumber = ST.StoreNumber
                AND SA.PID = P.PID
            GROUP BY ST.StoreNumber, TheYear, C.CityName
            ORDER BY TheYear ASC, Revenue DESC";

    $result = mysqli_query($conn, $sql);
    while ($rows = mysqli_fetch_array($result)) {

?>
    <tr>
        <td> <?php echo $rows['TheYear']; ?></td>
        <td> <?php echo $rows['StoreNumber']; ?></td>
        <td> <?php echo $rows['CityName']; ?></td>
        <td> <?php echo $rows['Revenue']; ?></td>
    </tr>

<?php
    }

    echo $sql;
?>
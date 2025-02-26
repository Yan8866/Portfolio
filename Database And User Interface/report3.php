<?php
session_start();
require 'includes/database.php';
$conn = getDB();
$cookieuserID = $_COOKIE['UserID'];

require 'includes/updateauditlog.php';

// get the full name of the current user 
$sql = "SELECT 
    -- Query the PID, PName, RetailPrice
    P.PID,
    P.PName,
    P.RetailPrice,
    -- Query all total number of units sold with the current product  
    COALESCE(SUM(S.Quantity), 0) AS Total_Units_Sold,
    
    -- Query total number of units sold at a discount
    (
        SELECT COALESCE(SUM(S2.Quantity), 0)
        FROM sale S2
        JOIN Discount D ON S2.PID = D.PID
                       AND S2.SaleDate = D.DiscountedDate
        WHERE P.PID = S2.PID
    ) AS Units_Sold_At_Discount_Price,
    
    -- Query total number of units sold at the retail price
    (
        SELECT COALESCE(SUM(S3.Quantity), 0)
        FROM sale S3
        LEFT JOIN discount D2 ON S3.PID = D2.PID
                             AND S3.SaleDate = D2.DiscountedDate
        WHERE P.PID = S3.PID AND D2.DiscountedDate IS NULL
    ) AS Units_Sold_At_Retail_Price,
    
    -- Query actual revenue
    ( ROUND(
        (SELECT COALESCE(SUM(S4.Quantity * D3.DiscountPrice), 0)
        FROM sale S4
        JOIN discount D3 ON S4.PID = D3.PID
                        AND S4.SaleDate = D3.DiscountedDate
        WHERE P.PID = S4.PID
        ) + (
        SELECT COALESCE(SUM(S5.Quantity * P.RetailPrice), 0)
        FROM sale S5
        LEFT JOIN discount D4 ON S5.PID = D4.PID
                             AND S5.SaleDate = D4.DiscountedDate
        WHERE P.PID = S5.PID AND D4.DiscountedDate IS NULL
        ), 2)
    ) AS Total_Actual_Revenue,

    -- Query predicted revenue
    ( ROUND(
        (SELECT COALESCE(SUM(S6.Quantity * 0.75 * P.RetailPrice), 0)
        FROM sale S6
        JOIN discount D5 ON S6.PID = D5.PID
                        AND S6.SaleDate = D5.DiscountedDate
        WHERE P.PID = S6.PID
        ) + (
        SELECT COALESCE(SUM(S7.Quantity * P.RetailPrice), 0)
        FROM sale S7
        LEFT JOIN discount D6 ON S7.PID = D6.PID
                             AND S7.SaleDate = D6.DiscountedDate
        WHERE P.PID = S7.PID AND D6.DiscountedDate IS NULL
        ), 2)
    ) AS Total_Predicted_Revenue,

    -- Query difference of actual and predicted revenues if the absolute value is larger than 200
    ( ROUND(
    ( (SELECT COALESCE(SUM(S8.Quantity * D7.DiscountPrice), 0)
            FROM sale S8
            JOIN discount D7 ON S8.PID = D7.PID
                            AND S8.SaleDate = D7.DiscountedDate
            WHERE P.PID = S8.PID
            ) + (
            SELECT COALESCE(SUM(S9.Quantity * P.RetailPrice), 0)
            FROM sale S9
            LEFT JOIN discount D8 ON S9.PID = D8.PID
                                AND S9.SaleDate = D8.DiscountedDate
            WHERE P.PID = S9.PID AND D8.DiscountedDate IS NULL
            )) -
            ((SELECT COALESCE(SUM(S10.Quantity * 0.75 * P.RetailPrice), 0)
            FROM sale S10
            JOIN discount D9 ON S10.PID = D9.PID
                             AND S10.SaleDate = D9.DiscountedDate
            WHERE P.PID = S10.PID
            ) +
            (SELECT COALESCE(SUM(S11.Quantity * P.RetailPrice), 0)
            FROM sale S11
            LEFT JOIN discount D10 ON S11.PID = D10.PID
                                  AND S11.SaleDate = D10.DiscountedDate
            WHERE P.PID = S11.PID AND D10.DiscountedDate IS NULL
            )), 2)
            ) AS Actual_Vs_Predicted_Rev_Difference
    
    FROM sale S
    LEFT JOIN product P ON P.PID = S.PID
    LEFT JOIN product_category PC ON P.PID = PC.PID
    WHERE PC.CategoryName = 'GPS'
    GROUP BY P.PID
    HAVING Actual_Vs_Predicted_Rev_Difference > 200 OR Actual_Vs_Predicted_Rev_Difference < -200
    ORDER BY Actual_Vs_Predicted_Rev_Difference DESC;";

$result = mysqli_query($conn, $sql);
$rows = mysqli_fetch_all($result, MYSQLI_ASSOC);
?>

<?php require 'includes/header.php'; ?>

<table>
  <caption> Report 3 - Actual vs Predicted Revenue for GPS units </caption> <br> <br>
  <thead>
    <tr>
      <th scope="col"> Product ID </th>
      <th scope="col"> Product Name </th>
      <th scope="col"> Retail Price </th>
      <th scope="col"> Total Number of Units Ever Sold </th>
      <th scope="col"> Total Number of Units Sold at a Discount </th>
      <th scope="col"> Total Number of Units Sold at Retail Price </th>
      <th scope="col"> Actual Revenue </th>
      <th scope="col"> Predicted Revenue </th>
      <th scope="col"> Difference </th>
    </tr>
  </thead>
  <tbody>
  <?php foreach ($rows as $row): ?>
    <tr>
      <th scope="row"> <?php echo $row["PID"] ?></th>
      <th scope="row"> <?php echo $row["PName"] ?></th>
      <td> <?php echo $row["RetailPrice"] ?></td>
      <td> <?php echo $row["Total_Units_Sold"] ?> </td>
      <td> <?php echo $row["Units_Sold_At_Discount_Price"] ?> </td>
      <td> <?php echo $row["Units_Sold_At_Retail_Price"] ?> </td>
      <td> <?php echo $row["Total_Actual_Revenue"] ?> </td>
      <td> <?php echo $row["Total_Predicted_Revenue"] ?> </td>
      <td> <?php echo $row["Actual_Vs_Predicted_Rev_Difference"] ?> </td>
    </tr>
  <?php endforeach; ?> 
</table>

<?php require 'includes/footer.php'; ?>
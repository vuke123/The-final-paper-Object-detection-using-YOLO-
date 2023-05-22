import csv

# Define the range for the loop
start = 1
end = 208

# Open the CSV file in write mode
with open('208examples.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(['image', 'text'])

    # Iterate over the range and generate rows
    for i in range(start, end+1):
        # Format the image and text values with leading zeros
        image = f"{i:06}.jpg"
        text = f"{i:06}.txt"

        # Write the row to the CSV file
        writer.writerow([image, text])

print("CSV file generated successfully!")

pip install -r requirements.txt

# download model
wget "https://disk.yandex.ru/d/t3pZDmLRHXHedg" -O best_model.zip
unzip best_model.zip
rm best_model.zip

# download custom test dataset
wget "https://disk.yandex.ru/d/QncL6hKiMCYVOQ" -O custom_test_data.zip
unzip custom_test_data.zip
rm custom_test_data.zip

# download test results
wget "https://disk.yandex.ru/d/6_VkJKqI2ffrMA" -O test_results.zip
unzip test_results.zip
rm test_results.zip
import telebot
import logic
import os

API_TOKEN = ''

bot = telebot.TeleBot(API_TOKEN)

# Проверяем необходимые файлы при запуске
missing_files = logic.check_required_files()
if missing_files:
    print(f"Ошибка: Отсутствуют необходимые файлы: {missing_files}")
    print("Убедитесь, что файлы находятся в той же папке!")
    exit()

# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, """\
Привет! Я бот для распознавания изображений. 
Просто отправь мне фото, и я попробую определить что на нем изображено.\
""")

# Обработка фотографий
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        # Скачиваем фото
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        file_name = f"temp_photo_{message.message_id}.jpg"
        
        # Сохраняем временный файл
        with open(file_name, 'wb') as new_file:
            new_file.write(downloaded_file)
        
        # Отправляем сообщение о начале обработки
        processing_msg = bot.reply_to(message, "🔄 Обрабатываю изображение...")
        
        # Обрабатываем изображение через ML модель
        class_name, confidence_score = logic.detect(file_name)
        
        # Форматируем проценты
        confidence_percent = round(confidence_score * 100, 2)
        
        # Отправляем результат
        result_text = f"Это: {class_name}\nТочность: {confidence_percent}%"
        bot.edit_message_text(
            chat_id=processing_msg.chat.id,
            message_id=processing_msg.message_id,
            text=result_text
        )
        
        # Удаляем временный файл
        if os.path.exists(file_name):
            os.remove(file_name)
    
    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка при обработке фото: {str(e)}")
        
        # Удаляем временный файл в случае ошибки
        if os.path.exists(file_name):
            os.remove(file_name)

# Handle all other messages
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    bot.reply_to(message, "Отправь мне фото для распознавания! 📷")

if __name__ == "__main__":
    print("Бот запущен...")

    bot.infinity_polling()

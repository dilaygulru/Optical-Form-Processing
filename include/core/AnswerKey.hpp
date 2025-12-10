#ifndef ANSWER_KEY_HPP
#define ANSWER_KEY_HPP

#include <string>
#include <vector>
#include <map>

class AnswerKey {
public:
    struct QuestionAnswer {
        std::string subject;      // Ders adı: turkce, matematik, vb.
        int questionNumber;       // Soru numarası (0'dan başlar)
        char correctAnswer;       // Doğru cevap: A, B, C, D, E
    };

    AnswerKey();
    
    // Cevap anahtarını yükle
    void loadAnswerKey(const std::vector<QuestionAnswer>& answers);
    
    // Öğrenci cevaplarını kontrol et ve puan hesapla
    struct ScoreResult {
        int totalQuestions;
        int correctAnswers;
        int wrongAnswers;
        int emptyAnswers;
        double score;
        std::map<std::string, int> subjectScores; // Ders bazında doğru sayısı
    };
    
    ScoreResult calculateScore(const std::map<std::string, std::string>& studentAnswers);
    
    // Tek bir cevabı kontrol et
    bool isCorrect(const std::string& subject, int questionIndex, char answer);
    
    // Doğru cevabı al
    char getCorrectAnswer(const std::string& subject, int questionIndex);

private:
    // subject -> (questionIndex -> correctAnswer)
    std::map<std::string, std::map<int, char>> answerKey_;
    
    // Öğrenci cevap stringini parse et (örn: "A,B,C,-,D" -> vector)
    std::vector<char> parseAnswerString(const std::string& answerStr);
};

#endif // ANSWER_KEY_HPP
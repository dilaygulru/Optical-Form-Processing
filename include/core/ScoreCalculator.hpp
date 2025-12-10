#ifndef SCORE_CALCULATOR_HPP
#define SCORE_CALCULATOR_HPP

#include "BubbleDetector.hpp"
#include "AnswerKey.hpp"
#include <map>

class ScoreCalculator {
public:
    ScoreCalculator(const AnswerKey& answerKey);
    
    // Tek bir ders için puanlama
    ScoreResult calculateScore(
        const std::string& subject,
        const std::vector<BubbleResult>& studentAnswers
    );
    
    // Tüm sınav için genel rapor
    struct ExamReport {
        std::map<std::string, ScoreResult> subjectScores;
        int totalCorrect;
        int totalWrong;
        int totalEmpty;
        double totalScore;
        double netScore;  // doğru - (yanlış/4)
    };
    
    ExamReport calculateFullExam(
        const std::map<std::string, std::vector<BubbleResult>>& allAnswers
    );
    
private:
    const AnswerKey& answerKey_;
};

#endif
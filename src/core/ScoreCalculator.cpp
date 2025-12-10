#include "ScoreCalculator.hpp"

ScoreCalculator::ScoreCalculator(const AnswerKey& answerKey)
    : answerKey_(answerKey) {}

ScoreResult ScoreCalculator::calculateScore(
    const std::string& subject,
    const std::vector<BubbleResult>& studentAnswers)
{
    ScoreResult result;
    result.subject = subject;
    result.correct = 0;
    result.wrong = 0;
    result.empty = 0;
    result.score = 0.0;
    result.details = studentAnswers;
    
    int questionCount = answerKey_.getQuestionCount(subject);
    
    for (size_t i = 0; i < studentAnswers.size() && i < (size_t)questionCount; ++i) {
        const auto& answer = studentAnswers[i];
        std::string correctAnswer = answerKey_.getCorrectAnswer(subject, i);
        
        if (answer.markedAnswer == "-" || !answer.isValid) {
            result.empty++;
        }
        else if (answer.markedAnswer == correctAnswer) {
            result.correct++;
        }
        else {
            result.wrong++;
        }
    }
    
    // Net hesaplama: doğru - (yanlış / 4)
    result.score = result.correct - (result.wrong / 4.0);
    
    return result;
}

ScoreCalculator::ExamReport ScoreCalculator::calculateFullExam(
    const std::map<std::string, std::vector<BubbleResult>>& allAnswers)
{
    ExamReport report;
    report.totalCorrect = 0;
    report.totalWrong = 0;
    report.totalEmpty = 0;
    report.totalScore = 0.0;
    
    for (const auto& [subject, answers] : allAnswers) {
        ScoreResult sr = calculateScore(subject, answers);
        report.subjectScores[subject] = sr;
        
        report.totalCorrect += sr.correct;
        report.totalWrong += sr.wrong;
        report.totalEmpty += sr.empty;
    }
    
    report.netScore = report.totalCorrect - (report.totalWrong / 4.0);
    report.totalScore = report.netScore;
    
    return report;
}
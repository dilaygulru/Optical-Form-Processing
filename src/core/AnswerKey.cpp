#include "core/AnswerKey.hpp"
#include <sstream>
#include <iostream>
#include <algorithm>

static std::vector<std::string> splitCSV(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t a = 0, b = tok.size();
        while (a < b && isspace((unsigned char)tok[a])) a++;
        while (b > a && isspace((unsigned char)tok[b - 1])) b--;
        out.push_back(tok.substr(a, b - a));
    }
    return out;
}

void AnswerKey::loadAnswerKey(const std::vector<QuestionAnswer>& keys) {
    keyMap_.clear();
    for (const auto& k : keys) {
        keyMap_[k.subject][k.questionNumber] = k.correctAnswer;
    }
}

AnswerKey::ScoreResult AnswerKey::calculateScore(
    const std::map<std::string, std::string>& studentAnswersCsv) 
{
    ScoreResult res;

    for (const auto& pair : keyMap_) {
        std::string subject = pair.first;
        const auto& correctMap = pair.second; 

        SubjectStat stat;
        
        std::string rawAnswers = "";
        if (studentAnswersCsv.find(subject) != studentAnswersCsv.end()) {
            rawAnswers = studentAnswersCsv.at(subject);
        }

        std::vector<std::string> tokens = splitCSV(rawAnswers);

        int maxQ = -1;
        for (const auto& q : correctMap) {
            if (q.first > maxQ) maxQ = q.first;
        }
        
        for (int qIdx = 0; qIdx <= maxQ; ++qIdx) {
            char correct = '-';
            if (correctMap.find(qIdx) != correctMap.end()) {
                correct = correctMap.at(qIdx);
            }

            char student = '-';
            if (qIdx < (int)tokens.size()) {
                std::string t = tokens[qIdx];
                if (!t.empty()) student = t[0]; 
            }

            if (student == 'X' || student == '-' || student == ' ' || student == '?') {
                stat.empty++;
            } 
            else if (student == correct) {
                stat.correct++;
            } 
            else {
                stat.wrong++;
            }
        }

        stat.net = stat.correct - (stat.wrong / 3.0);

        res.totalQuestions += (stat.correct + stat.wrong + stat.empty);
        res.totalCorrect += stat.correct;
        res.totalWrong += stat.wrong;
        res.totalEmpty += stat.empty;
        res.totalScore += stat.net;
        res.subjectDetails[subject] = stat;
    }

    return res;
}
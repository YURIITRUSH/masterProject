#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <math.h>

// ============================================================================
// 1. КОНФІГУРАЦІЯ СИСТЕМИ
// ============================================================================
namespace Config {
    constexpr unsigned long SERIAL_BAUD = 115200;

    // ─── ЗМІННІ ЧАСТОТИ (регулюються командами cfreq/tfreq) ───
    unsigned long control_interval_ms   = 10;   // 100 Гц за замовчуванням
    unsigned long telemetry_interval_ms = 10;   // 100 Гц за замовчуванням

    // Безпечні межі
    constexpr unsigned long CONTROL_MIN_MS  = 1;     // 1000 Гц максимум
    constexpr unsigned long CONTROL_MAX_MS  = 1000;  // 1 Гц мінімум
    constexpr unsigned long TELEM_MIN_MS    = 2;     // 500 Гц максимум (Serial-ліміт)
    constexpr unsigned long TELEM_MAX_MS    = 5000;  // 0.2 Гц мінімум

    // ─── Піни ───
    constexpr uint8_t PIN_MOTOR_ENA = 5;
    constexpr uint8_t PIN_MOTOR_IN1 = 7;
    constexpr uint8_t PIN_MOTOR_IN2 = 4;
    constexpr uint8_t PIN_ENC_A = 2;
    constexpr uint8_t PIN_ENC_B = 9;

    constexpr float TICKS_PER_REV = 2100.0f;

    constexpr uint8_t INA219_I2C_ADDR = 0x40;

    constexpr float MAX_VOLTAGE = 12.0f;

    // ─── ФІЛЬТРИ (time-based, щоб поведінка не залежала від частоти) ───
    constexpr float SPEED_FILTER_TAU   = 0.04f;  // стала часу 40 мс
    constexpr float CURRENT_FILTER_TAU = 0.03f;  // стала часу 30 мс

    // ─── Slew Rate Limiter (тепер у PWM/сек, не залежить від частоти) ───
    constexpr float SLEW_RATE_PWM_PER_SEC = 1000.0f; // 1000 ШІМ/сек
}

// ============================================================================
// 2. КЛАС КЕРУВАННЯ МОТОРОМ (L298N)
// ============================================================================
class MotorDriver {
public:
    enum class Direction { STOP, FORWARD, BACKWARD };

    MotorDriver(uint8_t pinPWM, uint8_t pinIn1, uint8_t pinIn2)
        : _pinPWM(pinPWM), _pinIn1(pinIn1), _pinIn2(pinIn2), _speed(0), _dir(Direction::STOP) {}

    void begin() {
        pinMode(_pinPWM, OUTPUT);
        pinMode(_pinIn1, OUTPUT);
        pinMode(_pinIn2, OUTPUT);
        stop();
    }

    void drive(Direction dir, int speed) {
        _speed = constrain(speed, 0, 255);
        _dir = (_speed == 0) ? Direction::STOP : dir;

        switch (_dir) {
            case Direction::FORWARD:
                digitalWrite(_pinIn1, HIGH);
                digitalWrite(_pinIn2, LOW);
                break;
            case Direction::BACKWARD:
                digitalWrite(_pinIn1, LOW);
                digitalWrite(_pinIn2, HIGH);
                break;
            case Direction::STOP:
                digitalWrite(_pinIn1, LOW);
                digitalWrite(_pinIn2, LOW);
                _speed = 0;
                break;
        }
        analogWrite(_pinPWM, _speed);
    }

    void stop() { drive(Direction::STOP, 0); }
    int getSpeed() const { return _speed; }
    Direction getDirection() const { return _dir; }
private:
    uint8_t _pinPWM, _pinIn1, _pinIn2;
    int _speed;
    Direction _dir;
};

// ============================================================================
// 3. КЛАС КВАДРАТУРНОГО ЕНКОДЕРА (з time-based фільтром та діагностикою)
// ============================================================================
class QuadratureEncoder {
public:
    QuadratureEncoder(uint8_t pinA, uint8_t pinB, float ticksPerRev)
        : _pinA(pinA), _pinB(pinB), _ticksPerRev(ticksPerRev), _ticks(0), _prevTicks(0), _omega(0.0f) {}

    void begin(void (*isrA)(), void (*isrB)()) {
        pinMode(_pinA, INPUT_PULLUP);
        pinMode(_pinB, INPUT_PULLUP);
        attachInterrupt(digitalPinToInterrupt(_pinA), isrA, CHANGE);
        attachInterrupt(digitalPinToInterrupt(_pinB), isrB, CHANGE);
    }

    inline void handleInterruptA() {
        _isrCount++;
        (digitalRead(_pinA) == digitalRead(_pinB)) ? _ticks-- : _ticks++;
    }

    inline void handleInterruptB() {
        _isrCount++;
        (digitalRead(_pinA) != digitalRead(_pinB)) ? _ticks-- : _ticks++;
    }

    void computeVelocity(float dt) {
        if (dt <= 0.001f) return;

        noInterrupts();
        long currentTicks = _ticks;
        interrupts();

        long deltaTicks = currentTicks - _prevTicks;
        _prevTicks = currentTicks;
        _lastDelta = deltaTicks;

        float revolutions = static_cast<float>(deltaTicks) / _ticksPerRev;
        float raw = (revolutions / dt) * 2.0f * PI;
        _lastRawOmega = raw;

        float alpha = 1.0f - expf(-dt / Config::SPEED_FILTER_TAU);
        alpha = constrain(alpha, 0.0f, 1.0f);

        _omega += alpha * (raw - _omega);
    }

    float getOmega() const { return _omega; }

    // ─── Діагностика ───
    long getRawTicks() const {
        noInterrupts();
        long t = _ticks;
        interrupts();
        return t;
    }
    unsigned long getIsrCount() const { return _isrCount; }
    long  getLastDelta() const { return _lastDelta; }
    float getRawOmega()  const { return _lastRawOmega; }

private:
    uint8_t _pinA, _pinB;
    float _ticksPerRev;
    volatile long _ticks;
    long _prevTicks;
    float _omega;

    volatile unsigned long _isrCount = 0;
    long _lastDelta = 0;
    float _lastRawOmega = 0.0f;
};

// ============================================================================
// 4. КЛАС МОНІТОРИНГУ ЖИВЛЕННЯ (INA219) — з узгодженням знаку для реверсу
// ============================================================================
class PowerMonitor {
public:
    PowerMonitor(uint8_t i2cAddr = 0x40)
        : _ina219(i2cAddr), _isOnline(false), _current(0.0f), _voltage(0.0f) {}

    bool begin() { _isOnline = _ina219.begin(); return _isOnline; }

    void update(MotorDriver::Direction dir, float dt) {
        if (!_isOnline || dt <= 0.001f) return;

        float raw_i = -_ina219.getCurrent_mA() / 1000.0f;
        float i = raw_i;
        if (dir == MotorDriver::Direction::BACKWARD) {
            i = -raw_i;
        }
        float v = _ina219.getBusVoltage_V();

        float alpha = 1.0f - expf(-dt / Config::CURRENT_FILTER_TAU);
        if (alpha > 1.0f) alpha = 1.0f;
        if (alpha < 0.0f) alpha = 0.0f;
        _current += alpha * (i - _current);
        _voltage = v;
    }

    float getCurrentAmps() const { return _current; }
    float getVoltageVolts() const { return _voltage; }
    bool isOnline() const { return _isOnline; }
private:
    Adafruit_INA219 _ina219;
    bool _isOnline;
    float _current, _voltage;
};

// ============================================================================
// 5. КЛАС PBC-РЕГУЛЯТОРА (Каскадний PID-PBC)
// ============================================================================
class PBCController {
public:
    volatile float Kp = 0.4f;
    volatile float Ki = 0.1f;
    volatile float Kd = 0.002f;
    float I_MAX = 2.0f;

    float R  = 4.5f;
    float ke = 0.6f;
    float Rd = 0.05f;

    float target_omega = 0.0f;
    bool  is_active = false;

    // Slew Rate Limiter — тепер float, щоб працювати при будь-якій частоті
    float current_pwm = 0.0f;

    // Скидання динаміки без чіпання current_pwm (для зміни цілі)
    void resetDynamics() {
        _z = 0.0f;
        _d_filtered = 0.0f;
    }

    // Нова ціль: зберегти поточний ШІМ, анти-D-kick через prev_e
    void beginTarget(float newTarget, float currentOmega, int currentPwmOut) {
        resetDynamics();
        _prev_e = newTarget - currentOmega;
        current_pwm = static_cast<float>(currentPwmOut);
        target_omega = newTarget;
        is_active = true;
    }

    // Повне скидання (тільки для STOP)
    void resetAll() {
        resetDynamics();
        _prev_e = 0.0f;
        current_pwm = 0.0f;
    }

    float computeVoltage(float current_I, float omega, float dt) {
        float e = target_omega - omega;

        // Інтеграл з інтегральним вікном
        if (fabsf(e) < 6.0f) {
            _z = constrain(_z + Ki * e * dt, -I_MAX, I_MAX);
        } else {
            _z *= expf(-dt / 0.3f);
        }

        // D-складова з фільтром
        float de_raw = (e - _prev_e) / dt;
        _prev_e = e;
        float d_alpha = 1.0f - expf(-dt / 0.02f);  // tau = 20 мс
        _d_filtered = _d_filtered + d_alpha * (de_raw - _d_filtered);

        // PID → i_ref
        float i_ref = _z + Kp * e + Kd * _d_filtered;
        i_ref = constrain(i_ref, -I_MAX, I_MAX);

        // Внутрішній PBC
        float u = ke * target_omega + R * i_ref - Rd * (current_I - i_ref);

        // Feedforward-компенсація тертя
        if (target_omega > 0.1f && u > 0.0f) {
            u += 0.3f + 0.02f * fabsf(target_omega);
        } else if (target_omega < -0.1f && u < 0.0f) {
            u -= 0.3f + 0.02f * fabsf(target_omega);
        }

        return constrain(u, -Config::MAX_VOLTAGE, Config::MAX_VOLTAGE);
    }
private:
    float _z = 0.0f;
    float _prev_e = 0.0f;
    float _d_filtered = 0.0f;
};

// ============================================================================
// 6. ОБРОБКА КОМАНД ТА ТЕЛЕМЕТРІЇ
// ============================================================================
class CommandController {
public:
    CommandController(MotorDriver& motor, QuadratureEncoder& encoder, PowerMonitor& power, PBCController& pbc)
        : _motor(motor), _encoder(encoder), _power(power), _pbc(pbc), _bufIdx(0) {}

    void pollSerial() {
        while (Serial.available() > 0) {
            char c = Serial.read();
            if (c == '\n' || c == '\r') {
                if (_bufIdx > 0) { _buffer[_bufIdx] = '\0'; parseAndExecute(_buffer); _bufIdx = 0; }
            } else if (_bufIdx < sizeof(_buffer) - 1) {
                _buffer[_bufIdx++] = tolower(c);
            }
        }
    }
    void turnTicks(long n, int pwm) {
        if (n == 0) return;
        pwm = constrain(pwm, 40, 255);

        _pbc.is_active = false;
        _pbc.resetAll();

        const long goal = labs(n);
        const long start = _encoder.getRawTicks();
        const MotorDriver::Direction dir =
            (n > 0) ? MotorDriver::Direction::FORWARD
                    : MotorDriver::Direction::BACKWARD;

        Serial.print(F("[TURN] start=")); Serial.print(start);
        Serial.print(F(" goal=")); Serial.print(goal);
        Serial.print(F(" pwm=")); Serial.println(pwm);

        const unsigned long t0 = millis();
        bool slow = false;
        _motor.drive(dir, pwm);

        while (true) {
            long done = labs(_encoder.getRawTicks() - start);
            if (done >= goal) break;

            if (millis() - t0 > 15000) {
                Serial.println(F("[TURN] Timeout (15 s)."));
                break;
            }
            if (millis() - t0 > 2000 && done == 0) {
                Serial.println(F("[TURN] Вал не рухнувся. Збільште ШІМ (напр. turn 4200 120)."));
                break;
            }
            // Останні 200 тіків — повільніше, щоб менше «пролетіти»
            if (!slow && (goal - done) < 200) {
                _motor.drive(dir, max(40, pwm * 2 / 3));
                slow = true;
            }
            delay(1);
        }

        _motor.stop();
        delay(500);   // дати валу докотитися

        const long endTicks = _encoder.getRawTicks();
        const long delta = endTicks - start;
        Serial.print(F("[TURN] end=")); Serial.print(endTicks);
        Serial.print(F(" delta=")); Serial.print(delta);
        Serial.print(F(" overshoot=")); Serial.println(labs(delta) - goal);
    }


    // Формат: Current_A,Voltage,Omega,Target,PWM,Ticks,dTicks
    void sendTelemetryCSV() {
        Serial.print(_power.getCurrentAmps(), 3); Serial.print(',');
        Serial.print(_power.getVoltageVolts(), 2); Serial.print(','); // <--- ДОДАНО НАПРУГУ
        Serial.print(_encoder.getOmega(), 2); Serial.print(',');
        Serial.print(_pbc.is_active ? _pbc.target_omega : 0.0f, 2); Serial.print(',');
        Serial.print(_motor.getSpeed()); Serial.print(',');
        Serial.print(_encoder.getRawTicks()); Serial.print(',');
        Serial.println(_encoder.getLastDelta());
    }

private:
    MotorDriver& _motor;
    QuadratureEncoder& _encoder;
    PowerMonitor& _power;
    PBCController& _pbc;
    char _buffer[64];
    uint8_t _bufIdx;

    void printStatus() {
        Serial.println(F("\n--- STATUS ---"));
        Serial.print(F("PBC Active:   ")); Serial.println(_pbc.is_active ? F("YES") : F("NO"));
        Serial.print(F("Target:       ")); Serial.print(_pbc.target_omega, 2); Serial.println(F(" rad/s"));
        Serial.print(F("Omega:        ")); Serial.print(_encoder.getOmega(), 2); Serial.println(F(" rad/s"));
        Serial.print(F("Voltage:      ")); Serial.print(_power.getVoltageVolts(), 2); Serial.println(F(" V"));
        Serial.print(F("INA219:       ")); Serial.println(_power.isOnline() ? F("ONLINE") : F("OFFLINE"));
        Serial.print(F("Current:      ")); Serial.print(_power.getCurrentAmps(), 3); Serial.println(F(" A"));
        Serial.print(F("PWM:          ")); Serial.println(_motor.getSpeed());
        Serial.print(F("ISR count:    ")); Serial.println(_encoder.getIsrCount());
        Serial.print(F("Raw ticks:    ")); Serial.println(_encoder.getRawTicks());
        Serial.print(F("Last dTicks:  ")); Serial.println(_encoder.getLastDelta());
        Serial.print(F("Raw omega:    ")); Serial.print(_encoder.getRawOmega(), 3); Serial.println(F(" rad/s"));
        Serial.print(F("Ctrl freq:    ")); Serial.print(1000.0f / Config::control_interval_ms, 1);
        Serial.print(F(" Hz (")); Serial.print(Config::control_interval_ms); Serial.println(F(" ms)"));
        Serial.print(F("Telem freq:   ")); Serial.print(1000.0f / Config::telemetry_interval_ms, 1);
        Serial.print(F(" Hz (")); Serial.print(Config::telemetry_interval_ms); Serial.println(F(" ms)"));
        Serial.print(F("Kp / Ki / Kd: ")); Serial.print(_pbc.Kp, 3);
        Serial.print(F(" / ")); Serial.print(_pbc.Ki, 3);
        Serial.print(F(" / ")); Serial.println(_pbc.Kd, 3);
        Serial.print(F("ke / R / Rd:  ")); Serial.print(_pbc.ke, 3);
        Serial.print(F(" / ")); Serial.print(_pbc.R, 2);
        Serial.print(F(" / ")); Serial.println(_pbc.Rd, 3);
        Serial.println(F("--------------\n"));
    }

    // Окремий рядок для швидкої діагностики (не матчить TELEMETRY_RE на ПК)
    void printDiag() {
        Serial.print(F("DIAG "));
        Serial.print(millis()); Serial.print(' ');
        Serial.print(_encoder.getIsrCount()); Serial.print(' ');
        Serial.print(_encoder.getRawTicks()); Serial.print(' ');
        Serial.print(_encoder.getLastDelta()); Serial.print(' ');
        Serial.println(_encoder.getOmega(), 3);
    }

    void parseAndExecute(char* cmdLine) {
        char* action = strtok(cmdLine, " ");
        if (!action) return;

        if (strcmp(action, "status") == 0) { printStatus(); return; }
        if (strcmp(action, "diag")   == 0) { printDiag();   return; }

        if (strcmp(action, "s") == 0 || strcmp(action, "stop") == 0) {
            _pbc.is_active = false;
            _pbc.target_omega = 0.0f;
            _pbc.resetAll();
            _motor.stop();
            return;
        }

        char* valStr = strtok(nullptr, " ");
        if (!valStr) return;
        float val = atof(valStr);
        
        if (strcmp(action, "turn") == 0) {
            char* pwmStr = strtok(nullptr, " ");
            int pwm = pwmStr ? atoi(pwmStr) : 80;
            turnTicks((long)val, pwm);
            return;
        }

        // ─── Зміна частоти КЕРУВАННЯ ───
        if (strcmp(action, "cfreq") == 0) {
            float hz = val;
            if (hz < 1.0f) hz = 1.0f;
            if (hz > 1000.0f) hz = 1000.0f;
            unsigned long ms = (unsigned long)(1000.0f / hz + 0.5f);
            ms = constrain(ms, Config::CONTROL_MIN_MS, Config::CONTROL_MAX_MS);
            Config::control_interval_ms = ms;
            Serial.print(F("[OK] Control freq = "));
            Serial.print(1000.0f / ms, 1);
            Serial.print(F(" Hz (")); Serial.print(ms); Serial.println(F(" ms)"));
            return;
        }

        // ─── Зміна частоти ТЕЛЕМЕТРІЇ ───
        if (strcmp(action, "tfreq") == 0) {
            float hz = val;
            if (hz < 0.2f) hz = 0.2f;
            if (hz > 500.0f) hz = 500.0f;
            unsigned long ms = (unsigned long)(1000.0f / hz + 0.5f);
            ms = constrain(ms, Config::TELEM_MIN_MS, Config::TELEM_MAX_MS);
            Config::telemetry_interval_ms = ms;
            Serial.print(F("[OK] Telemetry freq = "));
            Serial.print(1000.0f / ms, 1);
            Serial.print(F(" Hz (")); Serial.print(ms); Serial.println(F(" ms)"));
            return;
        }

        // ─── Коефіцієнти ───
        if (strcmp(action, "kp") == 0) { _pbc.Kp = val; Serial.print(F("[OK] Kp=")); Serial.println(val, 3); return; }
        if (strcmp(action, "ki") == 0) { _pbc.Ki = val; Serial.print(F("[OK] Ki=")); Serial.println(val, 3); return; }
        if (strcmp(action, "kd") == 0) { _pbc.Kd = val; Serial.print(F("[OK] Kd=")); Serial.println(val, 3); return; }
        if (strcmp(action, "rd") == 0) { _pbc.Rd = val; Serial.print(F("[OK] Rd=")); Serial.println(val, 3); return; }
        if (strcmp(action, "ke") == 0) { _pbc.ke = val; Serial.print(F("[OK] ke=")); Serial.println(val, 3); return; }
        if (strcmp(action, "r")  == 0) { _pbc.R  = val; Serial.print(F("[OK] R="));  Serial.println(val, 3); return; }

        // ─── Open-Loop ───
        if (strcmp(action, "f") == 0) {
            _pbc.is_active = false;
            _motor.drive(MotorDriver::Direction::FORWARD, (int)val);
        }
        else if (strcmp(action, "b") == 0) {
            _pbc.is_active = false;
            _motor.drive(MotorDriver::Direction::BACKWARD, (int)val);
        }
        // ─── PBC ───
        else if (strcmp(action, "w") == 0) {
            _pbc.beginTarget(val, _encoder.getOmega(), _motor.getSpeed());
        }
    }
};

// ============================================================================
// 7. ГОЛОВНИЙ ЦИКЛ ПРОГРАМИ
// ============================================================================
MotorDriver       motor(Config::PIN_MOTOR_ENA, Config::PIN_MOTOR_IN1, Config::PIN_MOTOR_IN2);
QuadratureEncoder encoder(Config::PIN_ENC_A, Config::PIN_ENC_B, Config::TICKS_PER_REV);
PowerMonitor      power(Config::INA219_I2C_ADDR);
PBCController     pbc;
CommandController cli(motor, encoder, power, pbc);

unsigned long lastTelemetryTime = 0;
unsigned long lastControlTime = 0;

void encoderISRA() { encoder.handleInterruptA(); }
void encoderISRB() { encoder.handleInterruptB(); }

void setup() {
    Serial.begin(Config::SERIAL_BAUD);
    motor.begin();

    // Перевірка, що піни енкодера мають переривання
    int irqA = (int)digitalPinToInterrupt(Config::PIN_ENC_A);
    int irqB = (int)digitalPinToInterrupt(Config::PIN_ENC_B);
    if (irqA < 0) {
        Serial.println(F("[FATAL] PIN_ENC_A does not support interrupts!"));
    }
    if (irqB < 0) {
        Serial.println(F("[FATAL] PIN_ENC_B does not support interrupts!"));
    }

    encoder.begin(encoderISRA, encoderISRB);

    power.begin();
    lastTelemetryTime = millis();
    lastControlTime = millis();
}

void loop() {
    cli.pollSerial();
    unsigned long currentTime = millis();

    // ─── Контур керування ───
    if (currentTime - lastControlTime >= Config::control_interval_ms) {
        float dt = (currentTime - lastControlTime) / 1000.0f;
        lastControlTime = currentTime;

        encoder.computeVelocity(dt);

        MotorDriver::Direction dir = pbc.is_active
            ? (pbc.target_omega >= 0 ? MotorDriver::Direction::FORWARD
                                     : MotorDriver::Direction::BACKWARD)
            : MotorDriver::Direction::STOP;
        power.update(dir, dt);

        if (pbc.is_active) {
            float u = pbc.computeVoltage(power.getCurrentAmps(), encoder.getOmega(), dt);
            float vSupply = (power.isOnline() && power.getVoltageVolts() > 1.0f)
                            ? power.getVoltageVolts() : Config::MAX_VOLTAGE;
            float duty = constrain(fabsf(u) / vSupply, 0.0f, 1.0f);
            float target_pwm = duty * 255.0f;

            // Slew Rate Limiter — time-based
            float max_delta = Config::SLEW_RATE_PWM_PER_SEC * dt;
            if (target_pwm > pbc.current_pwm + max_delta)       pbc.current_pwm += max_delta;
            else if (target_pwm < pbc.current_pwm - max_delta)  pbc.current_pwm -= max_delta;
            else                                                pbc.current_pwm = target_pwm;

            int pwm_out = (int)pbc.current_pwm;

            if (u > 0.01f) {
                motor.drive(MotorDriver::Direction::FORWARD, pwm_out);
            }
            else if (u < -0.01f) {
                motor.drive(MotorDriver::Direction::BACKWARD, pwm_out);
            }
            else {
                motor.stop();
                pbc.current_pwm = 0.0f;
            }
        }
    }

    // ─── Телеметрія ───
    if (currentTime - lastTelemetryTime >= Config::telemetry_interval_ms) {
        lastTelemetryTime = currentTime;
        cli.sendTelemetryCSV();
    }
}
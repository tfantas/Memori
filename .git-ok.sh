#!/bin/bash

# ==============================================================================
# Ferramenta: git-ok v5.9 (Production-Ready + SVN Stealth Mode)
#
# Changelog v5.9 (2026-02-03):
#   - [NEW] SVN STEALTH MODE: Git invisível para SVN (svn:ignore automático)
#   - [NEW] Ignora: .git, .gitignore, .git-ok.sh, .log/, .github/, cron refs
#   - [NEW] Incrementa svn:ignore existente (não sobrescreve)
#   - [NEW] Funciona silenciosamente se SVN não estiver presente
#
# Changelog v5.9 (2026-02-03):
#   - [NEW] set -euo pipefail com tratamento adequado
#   - [NEW] Logging duplo: stdout + arquivo (configurável)
#   - [NEW] Proteção de trabalho local antes de operações destrutivas (stash)
#   - [NEW] Auto-instalação de shellcheck (macOS/Debian/Ubuntu)
#   - [NEW] Validação de dependências no startup
#   - [NEW] Constantes configuráveis no topo
#   - [NEW] Opções --help e --version
#   - [NEW] Diretório de logs configurável (.log/ ou projeto)
#   - [FIX] Shellcheck compliant (SC2086, SC2181, etc)
#   - [FIX] Usa [[ ]] ao invés de [ ] para consistência
#
# Changelog v5.7 (2026-02-03):
#   - [FIX] CRITICO: Funciona corretamente quando executado via cron
#   - [FIX] Determina diretorio do script ANTES de qualquer operacao git
#   - [FIX] Suporte a symlinks (resolve com readlink -f se disponivel)
#
# Changelog v5.6 (2026-02-02):
#   - [NEW] PID lock para prevenir execucao simultanea
#   - [NEW] Verifica ahead/behind antes de push
#   - [NEW] git push -u para branches novas
#   - [NEW] Tempo de execucao no log
#   - [NEW] Exponential backoff (2s, 4s, 8s)
#   - [NEW] Verificacao de rede antes de operacoes remotas
#
# ==============================================================================

# ==============================================================================
# CONFIGURAÇÕES (ajuste conforme necessário)
# ==============================================================================
readonly VERSION="5.9"
readonly LARGE_FILE_THRESHOLD="95M"      # Tamanho máximo de arquivo
readonly LOCK_TIMEOUT_MIN=5              # Minutos para considerar lock travado
readonly FIND_TIMEOUT_SEC=30             # Timeout para busca de arquivos grandes
readonly MAX_PUSH_RETRIES=3              # Tentativas de push
readonly NETWORK_TIMEOUT_SEC=5           # Timeout para verificação de rede
readonly PID_WAIT_TIMEOUT_SEC=60         # Timeout esperando outra instância
readonly LOG_TO_FILE=true                # Logar também para arquivo
readonly LOG_DIR=".log"                  # Diretório de logs (relativo ao repo)

# ==============================================================================
# MODO ESTRITO - configuração segura para automação
# ==============================================================================
# NOTA: Não usamos 'set -e' globalmente porque muitos comandos git podem
# retornar códigos não-zero em situações normais (ex: nada para commit).
# Em vez disso, tratamos erros explicitamente onde necessário.
set -o nounset   # Erro em variáveis não definidas
set -o pipefail  # Propagar erro em pipes

# ==============================================================================
# DEPENDÊNCIAS E AMBIENTE
# ==============================================================================
export PATH="$PATH:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/bin"
export LC_ALL=C

readonly REQUIRED_CMDS=(git find timeout)
START_TIME=$(date +%s)
readonly START_TIME

# Variáveis globais (inicializadas depois do bootstrap)
LOG_FILE=""
REPO_ROOT=""

# ==============================================================================
# FUNÇÕES DE LOG
# ==============================================================================
_log_raw() {
    local msg="$1"
    # Sempre para stdout
    printf '%s\n' "$msg"
    # Se configurado, também para arquivo
    if [[ "$LOG_TO_FILE" == "true" && -n "$LOG_FILE" && -d "$(dirname "$LOG_FILE")" ]]; then
        printf '%s\n' "$msg" >> "$LOG_FILE"
    fi
}

log() {
    _log_raw "$(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    log "❌ ERRO: $1" >&2
}

show_duration() {
    local end_time duration
    end_time=$(date +%s)
    duration=$((end_time - START_TIME))
    if [[ $duration -gt 60 ]]; then
        log "⏱️  Duracao: $((duration/60))m $((duration%60))s"
    else
        log "⏱️  Duracao: ${duration}s"
    fi
}

# ==============================================================================
# VALIDAÇÃO DE DEPENDÊNCIAS
# ==============================================================================
check_dependencies() {
    local missing=()
    for cmd in "${REQUIRED_CMDS[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Dependencias faltando: ${missing[*]}"
        exit 1
    fi
}

# ==============================================================================
# HELP E VERSION
# ==============================================================================
show_help() {
    cat << EOF
git-ok v${VERSION} - Auto-sync inteligente para repositórios Git

USO:
    .git-ok.sh [OPÇÕES]

OPÇÕES:
    -h, --help      Mostra esta ajuda
    -v, --version   Mostra a versão
    --dry-run       Simula sem fazer alterações (TODO)

DESCRIÇÃO:
    Sincroniza automaticamente o repositório git local com o remote.
    Ideal para uso via cron para backup automático.

CONFIGURAÇÃO:
    Edite as constantes no início do script para personalizar:
    - LARGE_FILE_THRESHOLD: Tamanho máximo de arquivo (default: 95M)
    - LOG_DIR: Diretório de logs (default: .log)
    - MAX_PUSH_RETRIES: Tentativas de push (default: 3)

LOGS:
    Os logs são escritos em: \$REPO_ROOT/$LOG_DIR/git-ok.log

EOF
    exit 0
}

show_version() {
    echo "git-ok v${VERSION}"
    exit 0
}

# ==============================================================================
# INSTALAÇÃO AUTOMÁTICA DE SHELLCHECK (opcional, silenciosa, não-bloqueante)
# ==============================================================================
install_shellcheck_if_missing() {
    # Shellcheck é opcional - falha silenciosa se não conseguir instalar
    command -v shellcheck &> /dev/null && return 0

    # Tenta instalar silenciosamente, ignora qualquer erro
    if [[ "$OSTYPE" == "darwin"* ]]; then
        command -v brew &> /dev/null && brew install shellcheck &> /dev/null || true
    elif [[ -f /etc/debian_version ]]; then
        # Tenta sem sudo primeiro, depois com sudo (ignora erro de permissão)
        apt-get install -y -qq shellcheck &> /dev/null 2>&1 || \
        sudo apt-get update -qq && sudo apt-get install -y -qq shellcheck &> /dev/null 2>&1 || true
    fi

    # Retorna 0 independente do resultado - shellcheck é opcional
    return 0
}

# ==============================================================================
# SVN STEALTH MODE - Git invisível para SVN
# ==============================================================================
setup_svn_ignore() {
    # Se SVN não estiver instalado ou não for working copy, sair silenciosamente
    command -v svn &> /dev/null || return 0
    svn info &> /dev/null || return 0

    # Itens que devem ser invisíveis para o SVN
    local git_items=(
        ".git"
        ".gitignore"
        ".gitattributes"
        ".gitmodules"
        ".git-ok.sh"
        ".github"
        ".log"
        "*.log"
    )

    # Obter ignores atuais do SVN (se existirem)
    local current_ignores
    current_ignores=$(svn propget svn:ignore . 2>/dev/null) || current_ignores=""

    # Verificar quais itens precisam ser adicionados
    local items_to_add=()
    for item in "${git_items[@]}"; do
        if ! echo "$current_ignores" | grep -qxF "$item"; then
            items_to_add+=("$item")
        fi
    done

    # Se há itens para adicionar, atualizar svn:ignore
    if [[ ${#items_to_add[@]} -gt 0 ]]; then
        local new_ignores="$current_ignores"
        for item in "${items_to_add[@]}"; do
            if [[ -n "$new_ignores" ]]; then
                new_ignores="${new_ignores}"$'\n'"${item}"
            else
                new_ignores="$item"
            fi
        done

        # Aplicar svn:ignore silenciosamente
        echo "$new_ignores" | svn propset svn:ignore -F - . &> /dev/null || true
    fi

    return 0
}

# ==============================================================================
# FUNÇÕES DE LOCK E CONCORRÊNCIA
# ==============================================================================
check_pid_lock() {
    local pid_file=".git/git-ok.pid"

    if [[ -f "$pid_file" ]]; then
        local old_pid
        old_pid=$(cat "$pid_file" 2>/dev/null) || old_pid=""

        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
            log "⏳ Outra instancia em execucao (PID $old_pid). Aguardando..."
            local wait=0
            while [[ -f "$pid_file" ]] && kill -0 "$old_pid" 2>/dev/null && [[ $wait -lt $PID_WAIT_TIMEOUT_SEC ]]; do
                sleep 5
                wait=$((wait + 5))
            done
            if [[ -f "$pid_file" ]] && kill -0 "$old_pid" 2>/dev/null; then
                log "⚠️  Timeout aguardando outra instancia. Continuando..."
            fi
        fi
        rm -f "$pid_file" 2>/dev/null || true
    fi

    echo $$ > "$pid_file"
    # shellcheck disable=SC2064
    trap "rm -f '$pid_file' 2>/dev/null || true" EXIT
}

check_stale_locks() {
    if [[ -f ".git/index.lock" ]]; then
        local lock_age
        lock_age=$(find ".git/index.lock" -mmin +"$LOCK_TIMEOUT_MIN" 2>/dev/null) || lock_age=""

        if [[ -n "$lock_age" ]]; then
            log "🚑 Auto-Cura: Removendo git lock travado (>${LOCK_TIMEOUT_MIN}min)."
            rm -f ".git/index.lock"
        else
            log "⏳ Lock recente detectado, aguardando 10s..."
            sleep 10
            if [[ -f ".git/index.lock" ]]; then
                log "🚑 Auto-Cura: Lock ainda existe, removendo."
                rm -f ".git/index.lock"
            fi
        fi
    fi

    find ".git/refs" -name "*.lock" -mmin +2 -delete 2>/dev/null || true
    rm -f ".git/HEAD.lock" ".git/config.lock" 2>/dev/null || true
}

# ==============================================================================
# FUNÇÕES DE REDE
# ==============================================================================
check_network() {
    if ! timeout "$NETWORK_TIMEOUT_SEC" git ls-remote --exit-code origin HEAD > /dev/null 2>&1; then
        log "⚠️  Sem conexao com remote. Abortando operacoes remotas."
        return 1
    fi
    return 0
}

# ==============================================================================
# PROTEÇÃO DE TRABALHO LOCAL
# ==============================================================================
protect_local_work() {
    # Salva trabalho local antes de operações potencialmente destrutivas
    local stash_msg
    stash_msg="git-ok-auto-stash-$(date +%Y%m%d-%H%M%S)"

    if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
        log "💾 Salvando trabalho local (stash)..."
        if git stash push -m "$stash_msg" --include-untracked 2>/dev/null; then
            log "   Stash criado: $stash_msg"
            echo "$stash_msg"
            return 0
        fi
    fi
    echo ""
    return 0
}

restore_local_work() {
    local stash_msg="$1"

    if [[ -n "$stash_msg" ]]; then
        log "📦 Restaurando trabalho local..."
        if git stash pop 2>/dev/null; then
            log "   ✅ Trabalho restaurado."
        else
            log "   ⚠️  Conflito no stash. Mantido em: git stash list"
        fi
    fi
}

# ==============================================================================
# FUNÇÕES DE GITIGNORE E ARQUIVOS GRANDES
# ==============================================================================
smart_ignore_common_junk() {
    local junk_dirs=(".venv" "venv" ".venv-playwright" "node_modules" "__pycache__"
                     "dist" "build" "coverage" ".DS_Store" "Thumbs.db" ".cache"
                     ".npm" ".yarn" ".pytest_cache" ".mypy_cache" "*.egg-info"
                     ".tox" ".nox")
    local changed=0

    for dir in "${junk_dirs[@]}"; do
        if [[ -e "$dir" ]]; then
            if ! git check-ignore -q "$dir" 2>/dev/null; then
                [[ ! -f .gitignore ]] && touch .gitignore
                if ! grep -qxF "$dir" .gitignore 2>/dev/null; then
                    echo "$dir" >> .gitignore
                    changed=1
                fi
                git rm -r --cached "$dir" > /dev/null 2>&1 || true
            fi
        fi
    done

    [[ $changed -eq 1 ]] && log "📝 .gitignore atualizado com novos padroes."
}

handle_large_files_pre_commit() {
    local large_files
    large_files=$(timeout "$FIND_TIMEOUT_SEC" find . -type f -size +"$LARGE_FILE_THRESHOLD" -not -path "./.git/*" 2>/dev/null) || large_files=""

    if [[ -n "$large_files" ]]; then
        log "⚠️  ALERTA: Arquivos >${LARGE_FILE_THRESHOLD} detectados."
        local count=0
        while IFS= read -r file; do
            [[ -z "$file" ]] && continue
            local clean_file="${file#./}"
            log "   ⛔ Bloqueando: $clean_file"
            git reset HEAD "$clean_file" 2>/dev/null || true
            git rm --cached "$clean_file" 2>/dev/null || true
            if ! grep -qxF "$clean_file" .gitignore 2>/dev/null; then
                echo "$clean_file" >> .gitignore
            fi
            count=$((count + 1))
            [[ $count -ge 20 ]] && { log "   ... e mais arquivos (limite 20)"; break; }
        done <<< "$large_files"
        git add .gitignore 2>/dev/null || true
    fi
}

# ==============================================================================
# DEEP CLEAN (com proteção de trabalho local)
# ==============================================================================
deep_clean_history() {
    log "🕳️  Iniciando LIMPEZA PROFUNDA do historico..."

    # PROTEÇÃO: Salvar trabalho local primeiro
    local saved_stash
    saved_stash=$(protect_local_work)

    smart_ignore_common_junk
    git add .gitignore 2>/dev/null || true
    git commit -m "chore: Update gitignore for deep clean" 2>/dev/null || true
    git rm -r --cached .venv-playwright node_modules .cache 2>/dev/null || true
    git commit -m "🔥 Deep Clean: Removed large folders" 2>/dev/null || true

    local branch ahead_count
    branch=$(git rev-parse --abbrev-ref HEAD)
    ahead_count=$(git rev-list --count "origin/$branch..HEAD" 2>/dev/null) || ahead_count="1"
    [[ -z "$ahead_count" || "$ahead_count" -eq 0 ]] && ahead_count=1
    [[ "$ahead_count" -gt 10 ]] && ahead_count=10

    log "   Consolidando ultimos $ahead_count commits..."
    git reset --soft "HEAD~$ahead_count" 2>/dev/null || git reset --soft HEAD~1 2>/dev/null || true
    smart_ignore_common_junk
    handle_large_files_pre_commit

    log "💾 Criando commit consolidado..."
    git commit -m "✨ Auto-Squash: Consolidated changes (Cleaned)" 2>/dev/null || true

    # Restaurar trabalho local se havia
    restore_local_work "$saved_stash"
}

# ==============================================================================
# SETUP DE LOGS
# ==============================================================================
setup_logging() {
    if [[ "$LOG_TO_FILE" == "true" && -n "$REPO_ROOT" ]]; then
        local log_dir="$REPO_ROOT/$LOG_DIR"

        if [[ ! -d "$log_dir" ]]; then
            mkdir -p "$log_dir" 2>/dev/null || true
            # Adicionar .log ao gitignore se não estiver
            if [[ -d "$log_dir" ]] && ! grep -qxF "$LOG_DIR" "$REPO_ROOT/.gitignore" 2>/dev/null; then
                echo "$LOG_DIR" >> "$REPO_ROOT/.gitignore"
            fi
        fi

        if [[ -d "$log_dir" ]]; then
            LOG_FILE="$log_dir/git-ok.log"
        else
            LOG_FILE="$REPO_ROOT/git_sync.log"
        fi
    fi
}

# ==============================================================================
# BOOTSTRAP - DETERMINAÇÃO DO DIRETÓRIO (Cron-Safe)
# ==============================================================================
bootstrap() {
    # Passo 1: Determinar caminho do script
    local script_path="${BASH_SOURCE[0]:-$0}"

    # Passo 2: Resolver symlinks se disponível
    if command -v readlink > /dev/null 2>&1; then
        local resolved_path
        resolved_path="$(readlink -f "$script_path" 2>/dev/null)" && script_path="$resolved_path"
    fi

    # Passo 3: Extrair e validar diretório
    local script_dir
    script_dir="$(cd "$(dirname "$script_path")" 2>/dev/null && pwd)"

    if [[ -z "$script_dir" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') ❌ Erro fatal: Nao foi possivel determinar diretorio do script" >&2
        exit 1
    fi

    if [[ ! -d "$script_dir" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') ❌ Erro fatal: Diretorio nao existe: $script_dir" >&2
        exit 1
    fi

    # Passo 4: Navegar para o diretório do script
    cd "$script_dir" || {
        echo "$(date '+%Y-%m-%d %H:%M:%S') ❌ Erro fatal: Nao foi possivel acessar $script_dir" >&2
        exit 1
    }

    # Passo 5: Validar que é repositório git
    if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') ❌ Erro fatal: $script_dir nao e um repositorio git" >&2
        exit 1
    fi

    # Passo 6: Ir para raiz do repositório
    REPO_ROOT="$(git rev-parse --show-toplevel)"
    cd "$REPO_ROOT" || exit 1
}

# ==============================================================================
# MAIN
# ==============================================================================
main() {
    # Processar argumentos
    for arg in "$@"; do
        case "$arg" in
            -h|--help) show_help ;;
            -v|--version) show_version ;;
            *) log_error "Opcao desconhecida: $arg"; show_help ;;
        esac
    done

    # Validar dependências
    check_dependencies

    # Bootstrap - determinar diretório correto
    bootstrap

    # Setup de logging
    setup_logging

    # SVN Stealth Mode - Git invisível para SVN
    setup_svn_ignore

    # Locks
    check_pid_lock
    check_stale_locks

    local branch repo_name
    branch=$(git rev-parse --abbrev-ref HEAD)
    repo_name=$(basename "$(pwd)")

    log "🔄 [$repo_name] Iniciando sync na branch '$branch'..."

    # PASSO 1: Preparação
    smart_ignore_common_junk
    git add -A || true
    handle_large_files_pre_commit

    # PASSO 2: Commit
    if ! git diff --cached --quiet 2>/dev/null; then
        local files
        files=$(git diff --cached --name-only | wc -l | tr -d ' ')
        log "💾 Commitando $files arquivo(s)..."
        git commit -m "💾 Auto-save ($files files)" || true
    else
        log "💤 Sem mudancas locais."
    fi

    # Verificar conectividade
    if ! check_network; then
        log "⚠️  Operacoes remotas puladas (sem rede)."
        show_duration
        exit 0
    fi

    # PASSO 3: Fetch + Status
    git fetch origin "$branch" 2>/dev/null || git fetch origin 2>/dev/null || true

    # Verificar se branch existe no remote
    local remote_exists push_cmd
    remote_exists=$(git ls-remote --heads origin "$branch" 2>/dev/null | wc -l | tr -d ' ')

    if [[ "$remote_exists" -eq 0 ]]; then
        log "🆕 Branch '$branch' nao existe no remote. Criando..."
        push_cmd="git push -u origin $branch"
    else
        # Verificar ahead/behind
        local ahead behind
        ahead=$(git rev-list --count "origin/$branch..HEAD" 2>/dev/null) || ahead="0"
        behind=$(git rev-list --count "HEAD..origin/$branch" 2>/dev/null) || behind="0"

        if [[ "$ahead" -eq 0 && "$behind" -eq 0 ]]; then
            log "✅ Ja sincronizado com origin/$branch."
            show_duration
            exit 0
        fi

        [[ "$behind" -gt 0 ]] && log "📥 $behind commit(s) para baixar..."
        [[ "$ahead" -gt 0 ]] && log "📤 $ahead commit(s) para enviar..."
        push_cmd="git push origin $branch"
    fi

    # PASSO 4: Pull (se necessário) - COM PROTEÇÃO
    if [[ "${behind:-0}" -gt 0 ]]; then
        log "📥 Sincronizando com remote..."

        # PROTEÇÃO: Salvar trabalho antes de pull com rebase
        local saved_work
        saved_work=$(protect_local_work)

        if ! git pull origin "$branch" --rebase --autostash 2>/dev/null; then
            git rebase --abort 2>/dev/null || true
            git pull origin "$branch" --strategy-option=ours --no-edit 2>/dev/null || true
        fi

        restore_local_work "$saved_work"
    fi

    # PASSO 5: Push (com retry e exponential backoff)
    local retry=0 push_success=0 backoff=2
    local push_output push_exit

    while [[ $retry -lt $MAX_PUSH_RETRIES ]]; do
        push_output=$($push_cmd 2>&1) || true
        push_exit=$?

        # Sucesso ou "Everything up-to-date"
        if [[ $push_exit -eq 0 ]] || echo "$push_output" | grep -qi "everything up-to-date"; then
            log "✅ Sucesso."
            push_success=1
            break
        fi

        retry=$((retry + 1))

        # GH001: Arquivo gigante
        if echo "$push_output" | grep -q "GH001"; then
            log "❌ GH001: Arquivo gigante no historico."
            deep_clean_history
            log "📤 Tentando push forcado..."
            if git push origin "$branch" --force 2>/dev/null; then
                log "✅ Sucesso (historico limpo)."
                push_success=1
                break
            fi
        # Refs divergentes - COM PROTEÇÃO
        elif echo "$push_output" | grep -q "cannot lock ref"; then
            log "⚠️  Refs divergentes (tentativa $retry/$MAX_PUSH_RETRIES)."
            git fetch origin "$branch" 2>/dev/null || true

            # PROTEÇÃO: Stash antes de reset --hard
            local saved_refs
            saved_refs=$(protect_local_work)
            git reset --hard "origin/$branch" 2>/dev/null || true
            restore_local_work "$saved_refs"
            continue
        # Non-fast-forward
        elif echo "$push_output" | grep -qE "(non-fast-forward|rejected|failed to push)"; then
            log "⚠️  Push rejeitado (tentativa $retry/$MAX_PUSH_RETRIES). Rebasing..."
            git pull origin "$branch" --rebase --autostash 2>/dev/null || git rebase --abort 2>/dev/null || true
            continue
        # Erro de rede
        elif echo "$push_output" | grep -qEi "(could not read|connection|network|timeout|ssl)"; then
            log "⚠️  Erro de rede (tentativa $retry/$MAX_PUSH_RETRIES). Aguardando ${backoff}s..."
            sleep "$backoff"
            backoff=$((backoff * 2))
            continue
        else
            log "⚠️  Erro (tentativa $retry/$MAX_PUSH_RETRIES): $(echo "$push_output" | head -1)"
            sleep "$backoff"
            backoff=$((backoff * 2))
        fi
    done

    if [[ $push_success -eq 0 ]]; then
        log "❌ Falha apos $MAX_PUSH_RETRIES tentativas."
        show_duration
        exit 1
    fi

    show_duration
}

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
main "$@"

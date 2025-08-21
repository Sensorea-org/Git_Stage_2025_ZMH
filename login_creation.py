import bcrypt

# Ton mot de passe en clair
password = "admin01(EMP)"

# Générer un sel et hacher le mot de passe
hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

print("Mot de passe :", password)
print("Hash bcrypt :", hashed.decode())